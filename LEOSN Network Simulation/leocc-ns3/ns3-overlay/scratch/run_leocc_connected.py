#!/usr/bin/env python3
"""Run the Fig. 10/11-style LeoCC connected-phase baseline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from leocc_experiment_utils import run_ns3, summarize, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--warmup", type=float, default=20.0)
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "BP-generated",
            "ISL-generated",
            "BP-sigcomm",
            "ISL-sigcomm",
        ),
        default="all",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=None,
        help="Override packet error rate (default: 0.001)",
    )
    parser.add_argument("--jitter-seed", type=int, default=4101)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns3_dir = Path(__file__).resolve().parents[1]
    results_dir = ns3_dir / "results" / "connected"

    cases = [
        ("BP-generated", "generated", "BP", "BP-LeoCC", None),
        ("ISL-generated", "generated", "ISL", "ISL-LeoCC", None),
        ("BP-sigcomm", "sigcomm", "BP", "BP-LeoCC-SIGCOMM", None),
        ("ISL-sigcomm", "sigcomm", "ISL", "ISL-LeoCC-SIGCOMM", 100.0),
    ]
    all_cases = cases.copy()
    if args.case != "all":
        cases = [case for case in cases if case[0] == args.case]

    if not args.skip_build:
        subprocess.run(
            [str(ns3_dir / "ns3"), "build", "leocc-connected-eval"],
            cwd=ns3_dir,
            check=True,
        )

    summary_rows: list[dict[str, object]] = []
    for case_name, trace_set, path_mode, directory_name, target_min_rtt in cases:
        output_dir = results_dir / directory_name
        error_rate = args.error_rate
        if error_rate is None:
            error_rate = 0.001
        print(f"Running {case_name} -> {output_dir}", flush=True)
        ns3_args: dict[str, object] = {
                "traceSet": trace_set,
                "pathMode": path_mode,
                "stopTime": args.duration,
                "throughputPeriod": 0.5,
                "bandwidthJitterStd": 0.10,
                "jitterSeed": args.jitter_seed,
                "errorRate": error_rate,
                "queuePackets": 200,
                "deviceQueuePackets": 100,
                "outputDir": output_dir.relative_to(ns3_dir),
                "enableReconfiguration": "false",
                "enableHandover": "false",
            }
        if target_min_rtt is not None:
            ns3_args["targetMinRtt"] = target_min_rtt
        run_ns3(
            ns3_dir,
            ns3_args,
            output_dir,
        )
        metrics = summarize(output_dir, min(args.warmup, args.duration / 4.0))
        summary_rows.append(
            {
                "case": case_name,
                "dataset": trace_set,
                "path": path_mode,
                "duration_s": args.duration,
                "configured_error_rate": error_rate,
                **metrics,
            }
        )
        print(
            f"  throughput={metrics['mean_throughput_mbps']:.2f} Mbps, "
            f"utilization={metrics['utilization']:.3f}, "
            f"p95 RTT={metrics['p95_rtt_ms']:.2f} ms",
            flush=True,
        )

    summary_path = results_dir / "LeoCC-summary.csv"
    if args.case != "all" and summary_path.exists():
        import csv

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            previous_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("case") in {case[0] for case in all_cases}
            ]
        updated_cases = {str(row["case"]) for row in summary_rows}
        summary_rows.extend(row for row in previous_rows if row.get("case") not in updated_cases)
        case_order = {case[0]: index for index, case in enumerate(all_cases)}
        summary_rows.sort(key=lambda row: case_order.get(str(row.get("case")), len(case_order)))
    write_summary(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
