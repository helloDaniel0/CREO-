import itertools
import math
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, int(os.environ.get("CREO_TORCH_THREADS", "1"))))


VARIANTS = (
    "full",
    "no_dwt",
    "no_burst_pacing",
    "no_pdpa",
    "no_lstm",
    "no_cnn",
)

STATIC_ACTION_SPACE = [0.61, 0.85, 0.95, 1.0, 1.03, 1.27, 1.67]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_dwt: bool = True
    use_burst_pacing: bool = True
    use_pdpa: bool = True
    use_lstm: bool = True
    use_cnn: bool = True


def get_spec(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown ablation variant {name!r}; choose from {VARIANTS}")
    values = {
        "use_dwt": name != "no_dwt",
        "use_burst_pacing": name != "no_burst_pacing",
        "use_pdpa": name != "no_pdpa",
        "use_lstm": name != "no_lstm",
        "use_cnn": name != "no_cnn",
    }
    return AblationSpec(name=name, **values)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    window = max(1, min(int(window), values.size))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def daubechies_denoise(values):
    """Denoise a 1-D signal with periodic db2 DWT and soft thresholding."""
    original = np.asarray(values, dtype=np.float64)
    if original.size < 8:
        return original.astype(np.float32)

    target_len = 1 << int(math.ceil(math.log2(original.size)))
    padded = np.pad(original, (0, target_len - original.size), mode="edge")
    sqrt3 = math.sqrt(3.0)
    denom = 4.0 * math.sqrt(2.0)
    low = np.asarray(
        [1.0 + sqrt3, 3.0 + sqrt3, 3.0 - sqrt3, 1.0 - sqrt3],
        dtype=np.float64,
    ) / denom
    high = np.asarray([low[3], -low[2], low[1], -low[0]], dtype=np.float64)

    approximation = padded
    details = []
    while approximation.size >= 8:
        half = approximation.size // 2
        approx_next = np.empty(half, dtype=np.float64)
        detail = np.empty(half, dtype=np.float64)
        for index in range(half):
            sample = approximation[(2 * index + np.arange(4)) % approximation.size]
            approx_next[index] = np.dot(low, sample)
            detail[index] = np.dot(high, sample)
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * math.sqrt(2.0 * math.log(max(original.size, 2)))
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        details.append(detail)
        approximation = approx_next

    for detail in reversed(details):
        reconstructed = np.zeros(detail.size * 2, dtype=np.float64)
        for index in range(detail.size):
            positions = (2 * index + np.arange(4)) % reconstructed.size
            reconstructed[positions] += low * approximation[index] + high * detail[index]
        approximation = reconstructed
    return approximation[: original.size].astype(np.float32)


def read_capacity_trace(path, column=1):
    values = []
    with open(path, "r", encoding="utf-8") as trace_file:
        for line in trace_file:
            fields = line.split()
            if len(fields) <= column:
                continue
            try:
                value = float(fields[column])
            except ValueError:
                continue
            if math.isfinite(value) and value > 0.0:
                values.append(value)
    if len(values) < 8:
        raise ValueError(f"Capacity trace {path} has fewer than eight valid samples")
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class _PdpaSolution:
    negative: tuple
    positive: tuple
    coverage: frozenset
    coverage_probability: float
    cost: float


def _pdpa_score(negative, positive, support, probability, lag, tolerance):
    actions = tuple(negative) + tuple(positive)
    achievable = []
    for steps in range(1, lag):
        for sequence in itertools.product(actions, repeat=steps):
            achievable.append((sum(sequence), steps))

    covered = set()
    weighted_cost = 0.0
    for index, value in enumerate(support):
        matching_steps = [steps for total, steps in achievable if abs(total - value) <= tolerance]
        if matching_steps:
            covered.add(index)
            weighted_cost += probability[index] * min(matching_steps)
    coverage_probability = float(sum(probability[index] for index in covered))
    cost = weighted_cost / max(coverage_probability, 1e-12)
    return _PdpaSolution(
        negative=tuple(sorted(negative)),
        positive=tuple(sorted(positive)),
        coverage=frozenset(covered),
        coverage_probability=coverage_probability,
        cost=float(cost),
    )


def _dominates(left, right):
    no_worse = left.coverage.issuperset(right.coverage) and left.cost <= right.cost
    strictly_better = left.coverage != right.coverage or left.cost < right.cost
    return no_worse and strictly_better


def _pareto_prune(solutions, limit=256):
    unique = {}
    for solution in solutions:
        key = (solution.negative, solution.positive)
        previous = unique.get(key)
        if previous is None or solution.cost < previous.cost:
            unique[key] = solution

    frontier = []
    ordered = sorted(
        unique.values(),
        key=lambda item: (-item.coverage_probability, item.cost),
    )
    for candidate in ordered:
        if any(_dominates(existing, candidate) for existing in frontier):
            continue
        frontier = [existing for existing in frontier if not _dominates(candidate, existing)]
        frontier.append(candidate)
    return sorted(frontier, key=lambda item: (-item.coverage_probability, item.cost))[:limit]


def _candidate_bins(values, probability, sign, count=7, tolerance=0.025):
    indices = [index for index, value in enumerate(values) if value * sign > tolerance * 0.5]
    ranked = sorted(indices, key=lambda index: probability[index], reverse=True)
    selected = ranked[: max(3, count - 2)]
    if indices:
        selected.extend([min(indices, key=lambda index: values[index] * sign)])
        selected.extend([max(indices, key=lambda index: values[index] * sign)])
    result = {float(values[index]) for index in selected}
    step = tolerance
    while len(result) < 3:
        result.add(sign * step)
        step += tolerance
    result = sorted(result, key=abs)[:count]
    return sorted(result)


def pdpa_action_space(capacities, lag=3, tolerance=0.025, candidate_count=7):
    """Compute the three-up/three-down action set with Pareto DP."""
    capacities = np.asarray(capacities, dtype=np.float64)
    ratios = capacities[lag:] / np.maximum(capacities[:-lag], 1e-9)
    log_ratios = np.log(np.clip(ratios, 0.25, 4.0))
    quantized = np.round(log_ratios / tolerance) * tolerance
    support, counts = np.unique(quantized, return_counts=True)
    probability = counts.astype(np.float64) / counts.sum()
    negative_candidates = _candidate_bins(
        support, probability, -1, candidate_count, tolerance
    )
    positive_candidates = _candidate_bins(
        support, probability, 1, candidate_count, tolerance
    )

    frontier = [_PdpaSolution((), (), frozenset(), 0.0, 0.0)]
    frontier_sizes = []
    for _ in range(3):
        expanded = []
        for solution in frontier:
            for negative in negative_candidates:
                if negative in solution.negative:
                    continue
                for positive in positive_candidates:
                    if positive in solution.positive:
                        continue
                    expanded.append(
                        _pdpa_score(
                            solution.negative + (negative,),
                            solution.positive + (positive,),
                            support,
                            probability,
                            lag,
                            tolerance,
                        )
                    )
        frontier = _pareto_prune(expanded)
        frontier_sizes.append(len(frontier))
        if not frontier:
            raise RuntimeError("PDPA produced an empty Pareto frontier")

    selected = min(frontier, key=lambda item: (-item.coverage_probability, item.cost))
    negative = sorted(math.exp(value) for value in selected.negative)
    positive = sorted(math.exp(value) for value in selected.positive)
    actions = negative + [1.0] + positive
    metadata = {
        "lag": lag,
        "tolerance": tolerance,
        "ratio_samples": int(log_ratios.size),
        "support_bins": int(support.size),
        "negative_candidates": len(negative_candidates),
        "positive_candidates": len(positive_candidates),
        "frontier_sizes": frontier_sizes,
        "coverage_probability": selected.coverage_probability,
        "expected_combination_cost": selected.cost,
    }
    return actions, metadata


class AblationFeatureBuilder:
    def __init__(self, spec, capacity_window=300, history_len=10):
        self.spec = spec
        self.capacity_window = capacity_window
        self.history_len = history_len
        self.capacity_hist = deque(maxlen=capacity_window)
        self.metric_hist = deque(maxlen=history_len)
        self.last_rtt_ms = None
        self.last_action = 1.0
        self.min_rtt_ms = None
        self.scale_mbps = 1.0
        self.last_raw_capacity_mbps = 0.0
        self.last_denoised_capacity_mbps = 0.0
        self.last_true_capacity_mbps = 0.0

    @staticmethod
    def _pad_left(values, length):
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] >= length:
            return values[-length:]
        pad_shape = (length - values.shape[0],) + values.shape[1:]
        return np.concatenate([np.zeros(pad_shape, dtype=np.float32), values], axis=0)

    def make(self, obs, action_multiplier=None):
        obs = np.asarray(obs, dtype=np.float64)
        segment_size = max(float(obs[6]), 1.0)
        throughput_mbps = max(float(obs[15]) / 1e6, 0.0)
        sampled_capacity_mbps = max(float(obs[17]) / 1e6, 1e-6)
        true_capacity_mbps = (
            max(float(obs[19]) / 1e6, 1e-6) if obs.size > 19 else sampled_capacity_mbps
        )
        rtt_ms = max(float(obs[11]) / 1000.0, 1e-3)
        min_rtt_observed_ms = max(float(obs[12]) / 1000.0, 1e-3)
        loss_rate = max(float(obs[18]) / 1e6, 0.0)
        cwnd_packets = float(obs[5]) / segment_size
        inflight_packets = float(obs[8]) / segment_size
        bdp_packets = max(float(obs[16]), 1.0)

        self.scale_mbps = max(self.scale_mbps, sampled_capacity_mbps, throughput_mbps)
        self.min_rtt_ms = (
            min_rtt_observed_ms
            if self.min_rtt_ms is None
            else min(self.min_rtt_ms, min_rtt_observed_ms, rtt_ms)
        )
        rtt_gradient = 0.0 if self.last_rtt_ms is None else rtt_ms - self.last_rtt_ms
        self.last_rtt_ms = rtt_ms
        if action_multiplier is not None:
            self.last_action = float(action_multiplier)

        send_rate_proxy_mbps = 8e-3 * cwnd_packets * segment_size / max(rtt_ms, 1e-3)
        metric = np.asarray(
            [
                loss_rate,
                throughput_mbps / self.scale_mbps,
                sampled_capacity_mbps / self.scale_mbps,
                rtt_ms / max(self.min_rtt_ms, 1e-3),
                self.min_rtt_ms / 1000.0,
                rtt_gradient / 100.0,
                inflight_packets / bdp_packets,
                send_rate_proxy_mbps / self.scale_mbps,
                self.last_action,
            ],
            dtype=np.float32,
        )
        self.capacity_hist.append(sampled_capacity_mbps)
        self.metric_hist.append(metric)

        raw_capacity = np.asarray(self.capacity_hist, dtype=np.float32)
        capacity = daubechies_denoise(raw_capacity) if self.spec.use_dwt else raw_capacity.copy()
        trend = _moving_average(capacity, max(3, min(31, len(capacity) // 8)))
        fluctuation = capacity - trend

        self.last_raw_capacity_mbps = sampled_capacity_mbps
        self.last_denoised_capacity_mbps = float(capacity[-1])
        self.last_true_capacity_mbps = true_capacity_mbps
        scale = max(self.scale_mbps, 1e-6)
        return {
            "capacity": self._pad_left(capacity, self.capacity_window) / scale,
            "trend": self._pad_left(trend, self.capacity_window) / scale,
            "fluct": self._pad_left(fluctuation, self.capacity_window) / scale,
            "metrics": self._pad_left(np.asarray(self.metric_hist), self.history_len),
        }


class AblationNet(nn.Module):
    def __init__(self, action_dim, spec, history_len=10, metric_dim=9, hidden=96):
        super().__init__()
        self.spec = spec
        self.hidden = hidden
        if spec.use_lstm:
            self.cap_lstm = nn.LSTM(1, hidden, batch_first=True)
            self.trend_lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fluct_lstm = nn.LSTM(1, hidden, batch_first=True)
        if spec.use_cnn:
            self.metric_cnn = nn.Sequential(
                nn.Conv1d(metric_dim, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        fused = hidden * 3 + 64 + history_len * metric_dim
        self.shared = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.q1 = nn.Linear(128, action_dim)
        self.q2 = nn.Linear(128, action_dim)

    def encode(self, batch):
        metrics = batch["metrics"]
        batch_size = metrics.shape[0]
        if self.spec.use_lstm:
            _, (capacity_hidden, _) = self.cap_lstm(batch["capacity"].unsqueeze(-1))
            _, (trend_hidden, _) = self.trend_lstm(batch["trend"].unsqueeze(-1))
            _, (fluct_hidden, _) = self.fluct_lstm(batch["fluct"].unsqueeze(-1))
            temporal = [capacity_hidden[-1], trend_hidden[-1], fluct_hidden[-1]]
        else:
            temporal = [
                metrics.new_zeros((batch_size, self.hidden)),
                metrics.new_zeros((batch_size, self.hidden)),
                metrics.new_zeros((batch_size, self.hidden)),
            ]
        if self.spec.use_cnn:
            local = self.metric_cnn(metrics.transpose(1, 2)).squeeze(-1)
        else:
            local = metrics.new_zeros((batch_size, 64))
        fused = torch.cat(temporal + [local, metrics.flatten(start_dim=1)], dim=1)
        return self.shared(fused)

    def forward(self, batch):
        encoded = self.encode(batch)
        return self.actor(encoded), self.q1(encoded), self.q2(encoded)


@dataclass
class Transition:
    state: dict
    action: int
    reward: float
    next_state: dict
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, count):
        return random.sample(self.buffer, count)

    def __len__(self):
        return len(self.buffer)


class AblationAgent:
    def __init__(
        self,
        spec,
        action_space,
        capacity_window=300,
        history_len=10,
        replay_size=50000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4,
        device=None,
        eval_mode=False,
    ):
        self.spec = spec
        self.action_space = [float(value) for value in action_space]
        self.capacity_window = capacity_window
        self.history_len = history_len
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eval_mode = eval_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.builder = AblationFeatureBuilder(spec, capacity_window, history_len)
        self.replay = ReplayBuffer(replay_size)
        self.net = AblationNet(len(self.action_space), spec, history_len).to(self.device)
        self.target = AblationNet(len(self.action_space), spec, history_len).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = math.log(len(self.action_space)) * 0.7
        self.learn_steps = 0

    def observe(self, obs, action_multiplier=None):
        return self.builder.make(obs, action_multiplier)

    def _collate(self, states):
        return {
            key: torch.as_tensor(
                np.stack([state[key] for state in states]),
                dtype=torch.float32,
                device=self.device,
            )
            for key in ("capacity", "trend", "fluct", "metrics")
        }

    def choose_action(self, state, deterministic=False):
        with torch.no_grad():
            logits, _, _ = self.net(self._collate([state]))
            probabilities = torch.softmax(logits, dim=-1)
            if deterministic or self.eval_mode:
                index = int(torch.argmax(probabilities, dim=-1).item())
            else:
                index = int(torch.distributions.Categorical(probs=probabilities).sample().item())
        return index, self.action_space[index]

    @staticmethod
    def action_to_cwnd(obs, multiplier):
        obs = np.asarray(obs, dtype=np.float64)
        segment_size = max(int(obs[6]), 1)
        current_cwnd = max(float(obs[5]), 2.0 * segment_size)
        if not math.isfinite(current_cwnd):
            current_cwnd = 10.0 * segment_size
        return min(int(max(2 * segment_size, multiplier * current_cwnd)), 1 << 31)

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(Transition(state, action, reward, next_state, done))

    def learn(self):
        if self.eval_mode or len(self.replay) < self.batch_size:
            return None
        transitions = self.replay.sample(self.batch_size)
        states = self._collate([item.state for item in transitions])
        next_states = self._collate([item.next_state for item in transitions])
        actions = torch.as_tensor(
            [item.action for item in transitions], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        dones = torch.as_tensor(
            [item.done for item in transitions], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        logits, q1, q2 = self.net(states)
        with torch.no_grad():
            next_logits, target_q1, target_q2 = self.target(next_states)
            next_probability = torch.softmax(next_logits, dim=-1)
            next_log_probability = torch.log_softmax(next_logits, dim=-1)
            next_q = torch.min(target_q1, target_q2)
            next_value = (
                next_probability * (next_q - self.log_alpha.exp() * next_log_probability)
            ).sum(dim=1, keepdim=True)
            target = rewards + (1.0 - dones) * self.gamma * next_value

        q_loss = F.mse_loss(q1.gather(1, actions), target) + F.mse_loss(
            q2.gather(1, actions), target
        )
        probability = torch.softmax(logits, dim=-1)
        log_probability = torch.log_softmax(logits, dim=-1)
        actor_loss = (
            probability
            * (self.log_alpha.exp().detach() * log_probability - torch.min(q1, q2))
        ).sum(dim=1).mean()
        self.optimizer.zero_grad()
        (q_loss + actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()

        entropy = -(probability.detach() * log_probability.detach()).sum(dim=1).mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.net.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        self.learn_steps += 1
        return float(q_loss.detach().cpu()), float(actor_loss.detach().cpu())

    def save(self, path, pdpa_metadata=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "variant": self.spec.name,
                "action_space": self.action_space,
                "capacity_window": self.capacity_window,
                "history_len": self.history_len,
                "learn_steps": self.learn_steps,
                "pdpa": pdpa_metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None, eval_mode=True):
        checkpoint = torch.load(path, map_location=device or "cpu")
        spec = get_spec(checkpoint["variant"])
        agent = cls(
            spec=spec,
            action_space=checkpoint["action_space"],
            capacity_window=checkpoint["capacity_window"],
            history_len=checkpoint["history_len"],
            device=device,
            eval_mode=eval_mode,
        )
        agent.net.load_state_dict(checkpoint["net"])
        agent.target.load_state_dict(checkpoint.get("target", checkpoint["net"]))
        agent.learn_steps = checkpoint.get("learn_steps", 0)
        return agent, checkpoint.get("pdpa", {})


def ablation_reward(obs, loss_weight=0.2):
    obs = np.asarray(obs, dtype=np.float64)
    throughput = max(float(obs[15]), 0.0)
    sampled_capacity = max(float(obs[17]), throughput, 1.0)
    true_capacity = max(float(obs[19]), 1.0) if obs.size > 19 else sampled_capacity
    rtt = max(float(obs[11]), 1.0)
    min_rtt = max(float(obs[12]), 1.0)
    loss_rate = max(float(obs[18]) / 1e6, 0.0)
    reward = (throughput / true_capacity - loss_weight * loss_rate) / ((rtt / min_rtt) ** 2)
    return float(np.clip(reward, -2.0, 2.0)) if math.isfinite(reward) else 0.0
