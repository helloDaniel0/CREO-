#!/usr/bin/env python3
"""Shared runner and result metrics for the LeoCC ns-3 experiments."""

from __future__ import annotations

import bisect
import csv
import math
import statistics
import subprocess
from pathlib import Path


REQUIRED_RESULTS = (
    "throughput.dat",
    "realbw.dat",
    "rtt.dat",
    "prop.dat",
    "queueSize.dat",
    "cwnd.dat",
    "pacing.dat",
    "loss.dat",
)


def read_series(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                rows.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def values_after(rows: list[tuple[float, float]], warmup: float) -> list[float]:
    return [value for timestamp, value in rows if timestamp >= warmup and math.isfinite(value)]


def nearest_values(
    samples: list[tuple[float, float]], references: list[tuple[float, float]]
) -> list[float]:
    if not references:
        return [math.nan] * len(samples)
    times = [timestamp for timestamp, _ in references]
    values = [value for _, value in references]
    result: list[float] = []
    for timestamp, _ in samples:
        index = bisect.bisect_right(times, timestamp) - 1
        result.append(values[max(index, 0)])
    return result


def summarize(result_dir: Path, warmup: float) -> dict[str, float]:
    for filename in REQUIRED_RESULTS:
        path = result_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty result: {path}")

    throughput_rows = read_series(result_dir / "throughput.dat")
    capacity_rows = read_series(result_dir / "realbw.dat")
    rtt_rows = [(t, value) for t, value in read_series(result_dir / "rtt.dat") if t >= warmup]
    prop_rows = read_series(result_dir / "prop.dat")
    queue_rows = read_series(result_dir / "queueSize.dat")
    pacing_rows = read_series(result_dir / "pacing.dat")
    loss_rows = read_series(result_dir / "loss.dat")

    throughput = values_after(throughput_rows, warmup)
    capacity = values_after(capacity_rows, warmup)
    rtt = [value for _, value in rtt_rows]
    queue = values_after(queue_rows, warmup)
    pacing = values_after(pacing_rows, warmup)
    base_rtt = nearest_values(rtt_rows, prop_rows)
    queue_delay = [max(measured - base, 0.0) for measured, base in zip(rtt, base_rtt)]

    mean_throughput = statistics.fmean(throughput) if throughput else math.nan
    mean_capacity = statistics.fmean(capacity) if capacity else math.nan
    rtt_differences = [abs(second - first) for first, second in zip(rtt, rtt[1:])]

    return {
        "mean_throughput_mbps": mean_throughput,
        "mean_capacity_mbps": mean_capacity,
        "utilization": mean_throughput / mean_capacity if mean_capacity > 0 else math.nan,
        "throughput_std_mbps": statistics.pstdev(throughput) if len(throughput) > 1 else 0.0,
        "mean_rtt_ms": statistics.fmean(rtt) if rtt else math.nan,
        "p95_rtt_ms": percentile(rtt, 0.95),
        "p99_rtt_ms": percentile(rtt, 0.99),
        "p95_queue_delay_ms": percentile(queue_delay, 0.95),
        "mean_abs_rtt_jitter_ms": statistics.fmean(rtt_differences) if rtt_differences else 0.0,
        "p95_queue_packets": percentile(queue, 0.95),
        "pacing_std_mbps": statistics.pstdev(pacing) if len(pacing) > 1 else 0.0,
        "final_loss_rate": loss_rows[-1][1] if loss_rows else math.nan,
    }


def run_ns3(ns3_dir: Path, arguments: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    options = " ".join(f"--{key}={value}" for key, value in arguments.items())
    command = [str(ns3_dir / "ns3"), "run", f"leocc-connected-eval {options}"]
    completed = subprocess.run(
        command,
        cwd=ns3_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ns-3 failed for {output_dir} with status {completed.returncode}; see run.log"
        )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

