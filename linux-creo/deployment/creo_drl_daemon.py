#!/usr/bin/env python3
"""Run the CREO+ PyTorch policy against live Linux TCP telemetry."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import select
import signal
import struct
import sys
import time
from pathlib import Path

from creo_model_service import SharedModelService, default_checkpoint


ABI_VERSION = 1
ACTION_SOURCE_MODEL = 1
STATE_STRUCT = struct.Struct("<II" + "Q" * 7 + "IIi" + "I" * 8 + "Q" * 8)
ACTION_STRUCT = struct.Struct("<II" + "Q" * 4 + "I" * 4)
STATE_MESSAGE_SIZE = 172
ACTION_MESSAGE_SIZE = 56

if STATE_STRUCT.size != STATE_MESSAGE_SIZE or ACTION_STRUCT.size != ACTION_MESSAGE_SIZE:
    raise RuntimeError("CREO DRL ABI format has an unexpected size")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_state(payload: bytes) -> dict[str, object]:
    if len(payload) != STATE_MESSAGE_SIZE:
        raise RuntimeError(f"short CREO state message: {len(payload)} bytes")
    values = STATE_STRUCT.unpack(payload)
    if values[0] != ABI_VERSION or values[1] != STATE_MESSAGE_SIZE:
        raise RuntimeError(
            f"CREO ABI mismatch: version={values[0]} size={values[1]}"
        )
    return {
        "flow_id": values[2],
        "sequence": values[3],
        "timestamp_ns": values[4],
        "throughput_bps": values[5],
        "capacity_bps": values[6],
        "pacing_rate_bps": values[7],
        "last_action_sequence": values[8],
        "srtt_us": values[9],
        "min_rtt_us": values[10],
        "rtt_gradient_us": values[11],
        "loss_ppm": values[12],
        "inflight_pkts": values[13],
        "cwnd_pkts": values[14],
        "bdp_pkts": values[15],
        "app_limited": values[16],
        "last_action_q10": values[17],
        "last_action_source": values[18],
        "daemon_connected": values[19],
        "capacity_series_bps": list(values[20:28]),
    }


def model_request(state: dict[str, object]) -> dict[str, object]:
    return {
        "group_id": f"flow-{int(state['flow_id']):016x}",
        "telemetry": {
            "capacity_mbps": max(float(state["capacity_bps"]) / 1e6, 1e-3),
            "throughput_mbps": max(float(state["throughput_bps"]) / 1e6, 0.0),
            "rtt_ms": max(float(state["srtt_us"]) / 1000.0, 1e-3),
            "min_rtt_ms": max(float(state["min_rtt_us"]) / 1000.0, 1e-3),
            "loss_rate": float(state["loss_ppm"]) / 1e6,
            "inflight_packets": float(state["inflight_pkts"]),
            "bdp_packets": max(float(state["bdp_pkts"]), 1.0),
            "pacing_mbps": max(float(state["pacing_rate_bps"]) / 1e6, 0.0),
        },
    }


def encode_action(
    state: dict[str, object], response: dict[str, object], valid_for_ms: int
) -> bytes:
    return ACTION_STRUCT.pack(
        ABI_VERSION,
        ACTION_MESSAGE_SIZE,
        int(state["flow_id"]),
        int(state["sequence"]),
        time.monotonic_ns(),
        0,
        int(response["action_index"]),
        int(response["action_q10"]),
        0,
        valid_for_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=Path, default=Path("/dev/creo_drl"))
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument(
        "--state-dir", type=Path, default=Path("results/drl-online")
    )
    parser.add_argument("--capacity-trace", type=Path)
    parser.add_argument("--valid-for-ms", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-messages", type=int, default=0)
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if not args.device.exists():
        raise SystemExit(f"CREO kernel device does not exist: {args.device}")
    if not 50 <= args.valid_for_ms <= 5000:
        raise SystemExit("valid-for-ms must be in [50, 5000]")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "abi_version": ABI_VERSION,
        "pid": os.getpid(),
        "device": str(args.device),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "started_unix": time.time(),
    }
    (args.state_dir / "online-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    old_handlers = {
        signum: signal.signal(signum, stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    service = SharedModelService(
        args.checkpoint, args.state_dir, args.capacity_trace
    )
    pending: dict[tuple[int, int], int] = {}
    flow_ids: set[int] = set()
    states_received = 0
    actions_sent = 0
    kernel_model_actions = 0
    kernel_action_matches = 0
    kernel_action_mismatches = 0
    inference_samples: list[float] = []
    started = time.monotonic()
    log_path = args.state_dir / "online-control.jsonl"

    descriptor = os.open(
        args.device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC
    )
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLERR | select.POLLHUP)

    try:
        with log_path.open("a", encoding="utf-8") as action_log:
            while running:
                if args.stop_file and args.stop_file.exists():
                    break
                if args.duration and time.monotonic() - started >= args.duration:
                    break
                if args.max_messages and states_received >= args.max_messages:
                    break

                events = poller.poll(200)
                if not events:
                    continue
                if any(mask & (select.POLLERR | select.POLLHUP) for _, mask in events):
                    raise RuntimeError("CREO kernel device was closed")

                try:
                    payload = os.read(descriptor, STATE_MESSAGE_SIZE)
                except BlockingIOError:
                    continue
                state = decode_state(payload)
                states_received += 1
                flow_id = int(state["flow_id"])
                flow_ids.add(flow_id)

                acknowledged = None
                expected_q10 = None
                if int(state["last_action_source"]) == ACTION_SOURCE_MODEL:
                    kernel_model_actions += 1
                    key = (flow_id, int(state["last_action_sequence"]))
                    expected_q10 = pending.get(key)
                    acknowledged = expected_q10 == int(state["last_action_q10"])
                    if acknowledged:
                        kernel_action_matches += 1
                    else:
                        kernel_action_mismatches += 1

                response = service.infer([model_request(state)])[0]
                inference_samples.append(float(response["inference_us_per_group"]))
                action = encode_action(state, response, args.valid_for_ms)
                try:
                    written = os.write(descriptor, action)
                except OSError as error:
                    if error.errno == errno.ESTALE:
                        continue
                    raise
                if written != ACTION_MESSAGE_SIZE:
                    raise RuntimeError(f"short CREO action write: {written} bytes")

                actions_sent += 1
                pending[(flow_id, int(state["sequence"]))] = int(
                    response["action_q10"]
                )
                if len(pending) > 4096:
                    oldest = sorted(pending, key=lambda item: item[1])[:1024]
                    for key in oldest:
                        pending.pop(key, None)

                row = {
                    "time_unix": time.time(),
                    "flow_id": f"{flow_id:016x}",
                    "state_sequence": state["sequence"],
                    "throughput_mbps": float(state["throughput_bps"]) / 1e6,
                    "capacity_mbps": float(state["capacity_bps"]) / 1e6,
                    "rtt_ms": float(state["srtt_us"]) / 1000.0,
                    "min_rtt_ms": float(state["min_rtt_us"]) / 1000.0,
                    "loss_rate": float(state["loss_ppm"]) / 1e6,
                    "kernel_last_action_sequence": state["last_action_sequence"],
                    "kernel_last_action_q10": state["last_action_q10"],
                    "kernel_last_action_source": state["last_action_source"],
                    "expected_last_action_q10": expected_q10,
                    "kernel_acknowledged_model_action": acknowledged,
                    **response,
                }
                action_log.write(json.dumps(row, separators=(",", ":")) + "\n")
                action_log.flush()
    finally:
        os.close(descriptor)
        service.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.monotonic() - started
    summary = {
        **metadata,
        "elapsed_s": elapsed,
        "states_received": states_received,
        "actions_sent": actions_sent,
        "flow_ids": [f"{flow_id:016x}" for flow_id in sorted(flow_ids)],
        "kernel_model_actions_observed": kernel_model_actions,
        "kernel_action_matches": kernel_action_matches,
        "kernel_action_mismatches": kernel_action_mismatches,
        "closed_loop_verified": kernel_action_matches > 0
        and kernel_action_mismatches == 0,
        "mean_inference_us": (
            sum(inference_samples) / len(inference_samples)
            if inference_samples
            else None
        ),
        "finished_unix": time.time(),
    }
    (args.state_dir / "online-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["closed_loop_verified"] else 2


if __name__ == "__main__":
    sys.exit(main())
