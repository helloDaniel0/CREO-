#!/usr/bin/env python3
"""Generate and run targeted LeoCC weak-condition experiments."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from leocc_experiment_utils import run_ns3, summarize, write_summary


@dataclass(frozen=True)
class WeakCase:
    name: str
    path_mode: str
    capacity: Callable[[float, float], float]
    delay: Callable[[float, float], float]
    reconfiguration: bool
    interval: float = 15.0
    offset: float = 12.0


def constant_capacity(_: float, __: float) -> float:
    return 35.0


def constant_delay(_: float, __: float) -> float:
    return 2.0


def fast_oscillation(now: float, _: float) -> float:
    return 60.0 if int(now / 0.5) % 2 == 0 else 12.0


def abrupt_capacity(now: float, duration: float) -> float:
    fraction = now / duration
    if fraction < 0.25:
        return 55.0
    if fraction < 0.56:
        return 10.0
    if fraction < 0.75:
        return 50.0
    return 18.0


def soft_capacity(now: float, duration: float) -> float:
    fraction = now / duration
    if fraction < 0.21:
        return 40.0
    if fraction < 0.49:
        return 14.0
    if fraction < 0.76:
        return 55.0
    return 20.0


def soft_delay(now: float, duration: float) -> float:
    fraction = now / duration
    if fraction < 0.21:
        return 2.0
    if fraction < 0.49:
        return 10.0
    if fraction < 0.76:
        return 3.0
    return 8.0


def write_trace(trace_dir: Path, case: WeakCase, duration: float) -> tuple[Path, Path]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    bandwidth_path = trace_dir / f"{case.name}_bw.txt"
    latency_path = trace_dir / f"{case.name}_latency.txt"
    count = int(duration / 0.1) + 1

    bandwidth_lines: list[str] = []
    latency_lines: list[str] = []
    for index in range(count):
        now = index * 0.1
        capacity = case.capacity(now, duration)
        delay = case.delay(now, duration)
        bandwidth_lines.append(f"{index} {capacity:.6f} {2.0 * capacity:.6f}\n")
        latency_lines.append(f"{index} {delay:.6f} {delay:.6f}\n")

    bandwidth_path.write_text("".join(bandwidth_lines), encoding="utf-8")
    latency_path.write_text("".join(latency_lines), encoding="utf-8")
    return bandwidth_path, latency_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=80.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--case", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns3_dir = Path(__file__).resolve().parents[1]
    results_dir = ns3_dir / "results" / "leocc-weakness"
    trace_dir = results_dir / "traces"

    cases = [
        WeakCase("constant-control", "BP", constant_capacity, constant_delay, False),
        WeakCase(
            "false-positive-reconfiguration",
            "BP",
            constant_capacity,
            constant_delay,
            True,
            interval=2.0,
            offset=1.0,
        ),
        WeakCase("fast-capacity-oscillation", "BP", fast_oscillation, constant_delay, False),
        WeakCase("abrupt-unannounced-drop", "ISL", abrupt_capacity, constant_delay, False),
        WeakCase("soft-irregular-reconfiguration", "BP", soft_capacity, soft_delay, False),
    ]
    if args.case != "all":
        cases = [case for case in cases if case.name == args.case]
        if not cases:
            raise SystemExit(f"Unknown case: {args.case}")

    if not args.skip_build:
        subprocess.run(
            [str(ns3_dir / "ns3"), "build", "leocc-connected-eval"],
            cwd=ns3_dir,
            check=True,
        )

    rows: list[dict[str, object]] = []
    for case in cases:
        bandwidth_path, latency_path = write_trace(trace_dir, case, args.duration)
        output_dir = results_dir / case.name
        print(f"Running {case.name} -> {output_dir}", flush=True)
        run_ns3(
            ns3_dir,
            {
                "traceSet": "sigcomm",
                "pathMode": case.path_mode,
                "bwTrace": bandwidth_path.relative_to(ns3_dir),
                "latencyTrace": latency_path.relative_to(ns3_dir),
                "stopTime": args.duration,
                "outputDir": output_dir.relative_to(ns3_dir),
                "enableReconfiguration": str(case.reconfiguration).lower(),
                "reconfigurationInterval": case.interval,
                "reconfigurationOffset": case.offset,
            },
            output_dir,
        )
        metrics = summarize(output_dir, min(args.warmup, args.duration / 4.0))
        rows.append(
            {
                "case": case.name,
                "path": case.path_mode,
                "duration_s": args.duration,
                "reconfiguration_signal": case.reconfiguration,
                **metrics,
            }
        )
        print(
            f"  throughput={metrics['mean_throughput_mbps']:.2f} Mbps, "
            f"utilization={metrics['utilization']:.3f}, "
            f"p95 RTT={metrics['p95_rtt_ms']:.2f} ms, "
            f"RTT jitter={metrics['mean_abs_rtt_jitter_ms']:.3f} ms",
            flush=True,
        )

    summary_path = results_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
