#!/usr/bin/env python3
"""Runnable CREO+ shared inference and bounded fine-tuning prototype.

This is the control-plane half of the deployment described in
CREO_plus_deployment_response.pdf.  The kernel module remains the data-plane
executor.  Telemetry is grouped by terminal/path ID, model weights are shared,
and fine-tuning is asynchronous, coalesced, and bounded.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import queue
import random
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


torch.set_num_threads(max(1, int(os.environ.get("CREO_TORCH_THREADS", "1"))))
STATIC_ACTIONS = [0.61, 0.85, 0.95, 1.0, 1.03, 1.27, 1.67]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.full(window, 1.0 / window), mode="valid")


def db2_denoise(values: list[float] | np.ndarray) -> np.ndarray:
    """Periodic db2 DWT, MAD noise estimate, soft threshold, inverse DWT."""
    original = np.asarray(values, dtype=np.float64)
    if original.size < 8:
        return original.astype(np.float32)
    target_len = 1 << int(math.ceil(math.log2(original.size)))
    approximation = np.pad(original, (0, target_len - original.size), mode="edge")
    sqrt3 = math.sqrt(3.0)
    low = np.asarray(
        [1 + sqrt3, 3 + sqrt3, 3 - sqrt3, 1 - sqrt3], dtype=np.float64
    ) / (4 * math.sqrt(2.0))
    high = np.asarray([low[3], -low[2], low[1], -low[0]])
    details: list[np.ndarray] = []
    while approximation.size >= 8:
        half = approximation.size // 2
        next_approximation = np.empty(half)
        detail = np.empty(half)
        for index in range(half):
            samples = approximation[(2 * index + np.arange(4)) % approximation.size]
            next_approximation[index] = np.dot(low, samples)
            detail[index] = np.dot(high, samples)
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * math.sqrt(2.0 * math.log(max(original.size, 2)))
        details.append(np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0))
        approximation = next_approximation
    for detail in reversed(details):
        reconstructed = np.zeros(detail.size * 2)
        for index in range(detail.size):
            positions = (2 * index + np.arange(4)) % reconstructed.size
            reconstructed[positions] += low * approximation[index] + high * detail[index]
        approximation = reconstructed
    return approximation[: original.size].astype(np.float32)


def pdpa_action_space(capacities: list[float], lag: int = 3) -> tuple[list[float], dict[str, float]]:
    """Small Pareto search over three decrease/increase log-rate actions."""
    values = np.asarray(capacities, dtype=np.float64)
    if values.size <= lag + 8:
        return STATIC_ACTIONS.copy(), {"coverage": 0.0, "samples": int(values.size)}
    log_ratios = np.log(np.clip(values[lag:] / np.maximum(values[:-lag], 1e-6), 0.25, 4.0))
    negative = log_ratios[log_ratios < -0.01]
    positive = log_ratios[log_ratios > 0.01]
    if negative.size < 3 or positive.size < 3:
        return STATIC_ACTIONS.copy(), {"coverage": 0.0, "samples": int(log_ratios.size)}
    negative_candidates = sorted(set(np.quantile(negative, [0.1, 0.3, 0.5, 0.7, 0.9])))
    positive_candidates = sorted(set(np.quantile(positive, [0.1, 0.3, 0.5, 0.7, 0.9])))
    best: tuple[float, float, tuple[float, ...], tuple[float, ...]] | None = None
    tolerance = 0.035
    for down in itertools.combinations(negative_candidates, 3):
        for up in itertools.combinations(positive_candidates, 3):
            actions = down + up
            achievable = list(actions)
            achievable.extend(left + right for left in actions for right in actions)
            distances = np.min(
                np.abs(log_ratios[:, None] - np.asarray(achievable)[None, :]), axis=1
            )
            coverage = float(np.mean(distances <= tolerance))
            cost = float(np.mean(distances))
            candidate = (coverage, -cost, down, up)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    actions = sorted(math.exp(value) for value in best[2])
    actions += [1.0]
    actions += sorted(math.exp(value) for value in best[3])
    return actions, {
        "coverage": best[0],
        "mean_log_error": -best[1],
        "samples": int(log_ratios.size),
        "lag": lag,
    }


class SharedCreoNet(nn.Module):
    """Three temporal LSTMs plus a short cross-metric CNN."""

    def __init__(self, action_count: int, history_len: int = 10, metric_dim: int = 9):
        super().__init__()
        hidden = 96
        self.cap_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.trend_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fluct_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.metric_cnn = nn.Sequential(
            nn.Conv1d(metric_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden * 3 + 64 + history_len * metric_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_count)
        self.q1 = nn.Linear(128, action_count)
        self.q2 = nn.Linear(128, action_count)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        temporal = []
        for key, layer in (
            ("capacity", self.cap_lstm),
            ("trend", self.trend_lstm),
            ("fluct", self.fluct_lstm),
        ):
            _, (hidden, _) = layer(batch[key].unsqueeze(-1))
            temporal.append(hidden[-1])
        metrics = batch["metrics"]
        local = self.metric_cnn(metrics.transpose(1, 2)).squeeze(-1)
        encoded = self.shared(torch.cat(temporal + [local, metrics.flatten(1)], dim=1))
        return self.actor(encoded), self.q1(encoded), self.q2(encoded)


@dataclass
class PathContext:
    capacity_window: int
    history_len: int
    capacity: deque[float] = field(init=False)
    metrics: deque[np.ndarray] = field(init=False)
    min_rtt_ms: float = math.inf
    last_rtt_ms: float | None = None
    last_action: float = 1.0
    scale_mbps: float = 1.0
    previous_state: dict[str, np.ndarray] | None = None
    previous_action: int | None = None

    def __post_init__(self) -> None:
        self.capacity = deque(maxlen=self.capacity_window)
        self.metrics = deque(maxlen=self.history_len)

    @staticmethod
    def pad(values: np.ndarray, length: int) -> np.ndarray:
        if len(values) >= length:
            return values[-length:]
        shape = (length - len(values),) + values.shape[1:]
        return np.concatenate([np.zeros(shape, dtype=np.float32), values], axis=0)

    def build(self, telemetry: dict[str, float]) -> dict[str, np.ndarray]:
        capacity = max(float(telemetry["capacity_mbps"]), 1e-3)
        throughput = max(float(telemetry["throughput_mbps"]), 0.0)
        rtt = max(float(telemetry["rtt_ms"]), 1e-3)
        self.min_rtt_ms = min(self.min_rtt_ms, float(telemetry.get("min_rtt_ms", rtt)), rtt)
        gradient = 0.0 if self.last_rtt_ms is None else rtt - self.last_rtt_ms
        self.last_rtt_ms = rtt
        self.scale_mbps = max(self.scale_mbps, capacity, throughput)
        bdp = max(float(telemetry.get("bdp_packets", 1.0)), 1.0)
        metric = np.asarray(
            [
                float(telemetry.get("loss_rate", 0.0)),
                throughput / self.scale_mbps,
                capacity / self.scale_mbps,
                rtt / self.min_rtt_ms,
                self.min_rtt_ms / 1000.0,
                gradient / 100.0,
                float(telemetry.get("inflight_packets", 0.0)) / bdp,
                float(telemetry.get("pacing_mbps", throughput)) / self.scale_mbps,
                self.last_action,
            ],
            dtype=np.float32,
        )
        self.capacity.append(capacity)
        self.metrics.append(metric)
        raw = np.asarray(self.capacity, dtype=np.float32)
        denoised = db2_denoise(raw)
        trend = moving_average(denoised, max(3, min(31, len(denoised) // 8)))
        scale = self.scale_mbps
        return {
            "capacity": self.pad(denoised, self.capacity_window) / scale,
            "trend": self.pad(trend.astype(np.float32), self.capacity_window) / scale,
            "fluct": self.pad((denoised - trend).astype(np.float32), self.capacity_window) / scale,
            "metrics": self.pad(np.asarray(self.metrics), self.history_len),
        }


class BoundedReplayStore:
    def __init__(self, path: Path, per_group_limit: int = 2048):
        self.path = path
        self.limit = per_group_limit
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS transitions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT, created REAL, "
                "state TEXT, action INTEGER, reward REAL, next_state TEXT, done INTEGER)"
            )

    @staticmethod
    def encode(state: dict[str, np.ndarray]) -> str:
        return json.dumps({key: value.tolist() for key, value in state.items()}, separators=(",", ":"))

    @staticmethod
    def decode(payload: str) -> dict[str, np.ndarray]:
        return {key: np.asarray(value, dtype=np.float32) for key, value in json.loads(payload).items()}

    def add(
        self,
        group_id: str,
        state: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: dict[str, np.ndarray],
        done: bool = False,
    ) -> None:
        with self.lock, sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO transitions(group_id,created,state,action,reward,next_state,done) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    group_id,
                    time.time(),
                    self.encode(state),
                    action,
                    reward,
                    self.encode(next_state),
                    int(done),
                ),
            )
            database.execute(
                "DELETE FROM transitions WHERE group_id=? AND id NOT IN "
                "(SELECT id FROM transitions WHERE group_id=? ORDER BY id DESC LIMIT ?)",
                (group_id, group_id, self.limit),
            )

    def sample(self, count: int) -> list[tuple[dict[str, np.ndarray], int, float]]:
        with self.lock, sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT state,action,reward FROM transitions ORDER BY RANDOM() LIMIT ?", (count,)
            ).fetchall()
        return [(self.decode(state), int(action), float(reward)) for state, action, reward in rows]

    def count(self) -> int:
        with self.lock, sqlite3.connect(self.path) as database:
            return int(database.execute("SELECT COUNT(*) FROM transitions").fetchone()[0])


class SharedModelService:
    def __init__(
        self,
        checkpoint: Path | None,
        state_dir: Path,
        capacity_trace: Path | None = None,
    ):
        self.state_dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        self.capacity_window = 32
        self.history_len = 10
        self.actions = STATIC_ACTIONS.copy()
        payload = None
        if checkpoint and checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu")
            self.capacity_window = int(payload.get("capacity_window", self.capacity_window))
            self.history_len = int(payload.get("history_len", self.history_len))
            self.actions = [float(value) for value in payload.get("action_space", self.actions)]
        self.pdpa_metadata: dict[str, float] = {}
        if capacity_trace:
            capacities = []
            for line in capacity_trace.read_text(encoding="ascii").splitlines():
                try:
                    capacities.append(float(line.split()[1]))
                except (IndexError, ValueError):
                    continue
            self.actions, self.pdpa_metadata = pdpa_action_space(capacities)
        self.model = SharedCreoNet(len(self.actions), self.history_len)
        if payload:
            self.model.load_state_dict(payload["net"], strict=True)
        self.model.eval()
        self.model_lock = threading.RLock()
        self.contexts: dict[str, PathContext] = {}
        self.replay = BoundedReplayStore(state_dir / "replay.sqlite3")
        self.jobs: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._fine_tune_worker, daemon=True)
        self.worker.start()
        self.action_log = (state_dir / "actions.jsonl").open("a", encoding="utf-8")

    @staticmethod
    def batch(states: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(np.stack([state[key] for state in states]), dtype=torch.float32)
            for key in ("capacity", "trend", "fluct", "metrics")
        }

    def infer(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
        states = []
        contexts = []
        for request in requests:
            group_id = str(request.get("group_id", "default"))
            context = self.contexts.setdefault(
                group_id, PathContext(self.capacity_window, self.history_len)
            )
            telemetry = request["telemetry"]
            state = context.build(telemetry)
            if context.previous_state is not None and context.previous_action is not None:
                utilization = float(telemetry["throughput_mbps"]) / max(
                    float(telemetry["capacity_mbps"]), 1e-6
                )
                inflation = float(telemetry["rtt_ms"]) / context.min_rtt_ms
                reward = float(np.clip(utilization / (inflation * inflation), -2.0, 2.0))
                self.replay.add(
                    group_id,
                    context.previous_state,
                    context.previous_action,
                    reward,
                    state,
                )
            states.append(state)
            contexts.append((group_id, context))
        started = time.perf_counter_ns()
        with self.model_lock, torch.no_grad():
            logits, _, _ = self.model(self.batch(states))
            indices = torch.argmax(logits, dim=1).tolist()
        inference_us = (time.perf_counter_ns() - started) / 1000.0
        responses = []
        for (group_id, context), state, index in zip(contexts, states, indices):
            action = self.actions[index]
            context.previous_state = state
            context.previous_action = index
            context.last_action = action
            response = {
                "group_id": group_id,
                "action_index": index,
                "action_multiplier": action,
                "action_q10": round(action * 1024),
                "inference_us_per_group": inference_us / len(requests),
            }
            self.action_log.write(json.dumps({"time": time.time(), **response}) + "\n")
            responses.append(response)
        self.action_log.flush()
        if self.replay.count() >= 32 and self.replay.count() % 32 == 0:
            self.request_fine_tune("replay-threshold")
        return responses

    def request_fine_tune(self, reason: str) -> bool:
        try:
            self.jobs.put_nowait(reason)
            return True
        except queue.Full:
            return False

    def _fine_tune_worker(self) -> None:
        while not self.stop.is_set():
            try:
                reason = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if reason is None:
                self.jobs.task_done()
                return
            samples = self.replay.sample(32)
            if len(samples) >= 8:
                with self.model_lock:
                    candidate = copy.deepcopy(self.model).train()
                optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-5)
                states = self.batch([sample[0] for sample in samples])
                actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
                rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float32)
                logits, _, _ = candidate(states)
                weights = torch.softmax(rewards, dim=0) * len(rewards)
                loss = (functional.cross_entropy(logits, actions, reduction="none") * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5.0)
                optimizer.step()
                candidate.eval()
                temporary = self.state_dir / "stable-model.pt.tmp"
                stable = self.state_dir / "stable-model.pt"
                torch.save(
                    {
                        "net": candidate.state_dict(),
                        "action_space": self.actions,
                        "capacity_window": self.capacity_window,
                        "history_len": self.history_len,
                        "fine_tune_reason": reason,
                        "fine_tune_loss": float(loss.detach()),
                        "created": time.time(),
                    },
                    temporary,
                )
                os.replace(temporary, stable)
                with self.model_lock:
                    self.model.load_state_dict(candidate.state_dict())
                    self.model.eval()
            self.jobs.task_done()

    def close(self) -> None:
        self.stop.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=5)
        self.action_log.close()


def default_checkpoint() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "ns3-creo/ns3-overlay/contrib/ai/examples/rl-tcp/use-gym/checkpoints/example-smoke.pt"
    )


def self_test(service: SharedModelService, output: Path) -> dict[str, object]:
    random.seed(7)
    last_response = None
    for step in range(48):
        capacity = 26.0 + 7.0 * math.sin(step / 5.0)
        throughput = capacity * (0.82 + 0.03 * math.sin(step / 3.0))
        telemetry = {
            "capacity_mbps": capacity,
            "throughput_mbps": throughput,
            "rtt_ms": 20.0 * (1.0 + max(throughput / capacity - 0.8, 0.0)),
            "min_rtt_ms": 20.0,
            "loss_rate": 0.001,
            "inflight_packets": 45.0,
            "bdp_packets": 50.0,
            "pacing_mbps": throughput,
        }
        last_response = service.infer([{"group_id": "dish-001", "telemetry": telemetry}])[0]
    service.request_fine_tune("self-test")
    service.jobs.join()
    summary = {
        "status": "ok",
        "requests": 48,
        "path_groups": len(service.contexts),
        "replay_rows": service.replay.count(),
        "last_response": last_response,
        "actions": service.actions,
        "pdpa": service.pdpa_metadata,
        "stable_model_written": (output / "stable-model.pt").exists(),
    }
    (output / "self-test-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--state-dir", type=Path, default=Path("results/deployment-prototype"))
    parser.add_argument("--capacity-trace", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = SharedModelService(args.checkpoint, args.state_dir, args.capacity_trace)
    try:
        if args.self_test:
            print(json.dumps(self_test(service, args.state_dir), indent=2, sort_keys=True))
            return 0
        for line in sys.stdin:
            request = json.loads(line)
            requests = request if isinstance(request, list) else [request]
            print(json.dumps(service.infer(requests)), flush=True)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
