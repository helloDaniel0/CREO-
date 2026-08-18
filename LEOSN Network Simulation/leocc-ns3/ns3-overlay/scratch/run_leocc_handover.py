#!/usr/bin/env python3
"""Run and assess the Fig. 19/20-style LeoCC handover phase."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from pathlib import Path

from leocc_experiment_utils import percentile, read_series, run_ns3, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--case", choices=("BP", "ISL", "all"), default="BP")
    parser.add_argument("--handover-time", type=float, default=15.0)
    parser.add_argument("--handover-duration-ms", type=float, default=50.0)
    parser.add_argument("--jitter-seed", type=int, default=4101)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def values_between(
    rows: list[tuple[float, float]], start: float, stop: float
) -> list[float]:
    return [value for timestamp, value in rows if start <= timestamp < stop]


def binned_percentile(
    rows: list[tuple[float, float]], end: float, period: float, fraction: float
) -> float:
    values = values_between(rows, end - period, end)
    return percentile(values, fraction)


def first_mode_after(
    rows: list[tuple[float, float]], start: float, mode: int
) -> float:
    return next(
        (timestamp for timestamp, value in rows if timestamp >= start and int(value) == mode),
        math.nan,
    )


def first_mode_change_after(
    rows: list[tuple[float, float]], start: float, mode: int
) -> float:
    return next(
        (timestamp for timestamp, value in rows if timestamp > start and int(value) != mode),
        math.nan,
    )


def recovery_time(
    throughput: list[tuple[float, float]],
    rtt: list[tuple[float, float]],
    handover_time: float,
    duration_ms: float,
    sample_period: float,
) -> tuple[float, float, float]:
    pre_start = handover_time - 0.6
    pre_stop = handover_time
    pre_rates = values_between(throughput, pre_start, pre_stop)
    pre_rtt = values_between(rtt, pre_start, pre_stop)
    baseline_rate = statistics.median(pre_rates)
    baseline_rtt_p95 = percentile(pre_rtt, 0.95)
    rate_threshold = 0.80 * baseline_rate
    rtt_threshold = baseline_rtt_p95 + 15.0
    outage_end = handover_time + duration_ms / 1000.0

    candidates = [
        (timestamp, value)
        for timestamp, value in throughput
        if timestamp >= outage_end + sample_period
    ]
    for index in range(max(0, len(candidates) - 2)):
        window = candidates[index : index + 3]
        if any(rate < rate_threshold for _, rate in window):
            continue
        if any(
            not math.isfinite(binned_percentile(rtt, timestamp, sample_period, 0.95))
            or binned_percentile(rtt, timestamp, sample_period, 0.95) > rtt_threshold
            for timestamp, _ in window
        ):
            continue
        return window[0][0] - handover_time, baseline_rate, baseline_rtt_p95
    return math.nan, baseline_rate, baseline_rtt_p95


def analyze_case(
    algorithm: str,
    result_dir: Path,
    throughput_file: str,
    handover_time: float,
    duration_ms: float,
    sample_period: float,
) -> dict[str, object]:
    throughput = read_series(result_dir / throughput_file)
    rtt = read_series(result_dir / "rtt.dat")
    queue_path = result_dir / "queueSize.dat"
    queue = read_series(queue_path) if queue_path.exists() else []
    loss_path = result_dir / "loss.dat"
    loss = read_series(loss_path) if loss_path.exists() else []
    mode_path = result_dir / "leocc_mode.dat"
    mode = read_series(mode_path) if mode_path.exists() else []

    recovered, baseline_rate, baseline_rtt = recovery_time(
        throughput,
        rtt,
        handover_time,
        duration_ms,
        sample_period,
    )
    adaptation_start = first_mode_after(mode, handover_time, 3) if mode else math.nan
    adaptation_end = (
        first_mode_change_after(mode, adaptation_start, 3)
        if mode and math.isfinite(adaptation_start)
        else math.nan
    )
    immediate_pre_rtt = values_between(
        rtt, handover_time - sample_period, handover_time
    )
    post_handover_rtt = values_between(
        rtt, handover_time, handover_time + 1.0
    )
    immediate_pre_rtt_median = (
        statistics.median(immediate_pre_rtt) if immediate_pre_rtt else math.nan
    )
    post_handover_peak_rtt = max(post_handover_rtt, default=math.nan)
    tx_at_handover = next(
        (value for timestamp, value in throughput if timestamp >= handover_time),
        math.nan,
    )
    return {
        "algorithm": algorithm,
        "result_directory": str(result_dir),
        "throughput_source": throughput_file,
        "handover_time_s": handover_time,
        "interruption_ms": duration_ms,
        "pre_handover_throughput_mbps": baseline_rate,
        "pre_handover_p95_rtt_ms": baseline_rtt,
        "recovery_time_s": recovered,
        "tx_at_handover_mbps": tx_at_handover,
        "minimum_tx_15_16_mbps": min(
            values_between(throughput, handover_time, handover_time + 1.0),
            default=math.nan,
        ),
        "immediate_pre_handover_rtt_median_ms": immediate_pre_rtt_median,
        "peak_rtt_15_16_ms": post_handover_peak_rtt,
        "post_handover_rtt_increase_ms": (
            post_handover_peak_rtt - immediate_pre_rtt_median
            if math.isfinite(post_handover_peak_rtt)
            and math.isfinite(immediate_pre_rtt_median)
            else math.nan
        ),
        "mean_throughput_15_5_16_mbps": statistics.fmean(
            values_between(throughput, handover_time + 0.5, handover_time + 1.0)
        ),
        "peak_queue_15_16_packets": max(
            values_between(queue, handover_time, handover_time + 1.0), default=math.nan
        ),
        "final_loss_rate": loss[-1][1] if loss else math.nan,
        "reconfiguration_adaptation_start_s": adaptation_start,
        "reconfiguration_adaptation_end_s": adaptation_end,
    }


def write_acceptance(path: Path, rows: list[dict[str, object]]) -> None:
    by_name = {str(row["algorithm"]): row for row in rows}
    leo = by_name.get("LeoCC-BP") or by_name.get("LeoCC-ISL")
    if leo is None:
        raise RuntimeError("No LeoCC handover result was available for assessment")
    creo_plus = by_name.get("CREO+")
    creo = by_name.get("CREO")
    lines = [
        "LeoCC Fig. 19/20 placement check",
        "================================",
        "Recovery requires three consecutive 50 ms bins with Tx throughput >=80% of "
        "the pre-handover median and RTT p95 <= pre-handover p95 + 15 ms.",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['algorithm']}: recovery={float(row['recovery_time_s']):.3f}s, "
            f"peakRTT={float(row['peak_rtt_15_16_ms']):.2f}ms, "
            f"RTTincrease={float(row.get('post_handover_rtt_increase_ms', math.nan)):.2f}ms, "
            f"postTx={float(row['mean_throughput_15_5_16_mbps']):.2f}Mbps"
        )
    if creo_plus and creo:
        leo_recovery = float(leo["recovery_time_s"])
        plus_recovery = float(creo_plus["recovery_time_s"])
        creo_recovery = float(creo["recovery_time_s"])
        placement = (
            math.isfinite(leo_recovery)
            and math.isfinite(plus_recovery)
            and math.isfinite(creo_recovery)
            and plus_recovery <= leo_recovery <= creo_recovery + 0.10
        )
        reactive_detection = (
            float(leo["tx_at_handover_mbps"]) > 0.0
            and float(leo["reconfiguration_adaptation_start_s"])
            >= float(leo["handover_time_s"])
            + float(leo["interruption_ms"]) / 1000.0
        )
        bp = by_name.get("LeoCC-BP")
        isl = by_name.get("LeoCC-ISL")
        high_bdp_cost = bool(
            bp
            and isl
            and float(isl["recovery_time_s"]) >= float(bp["recovery_time_s"])
            and float(isl["peak_queue_15_16_packets"])
            > float(bp["peak_queue_15_16_packets"])
            and float(isl["post_handover_rtt_increase_ms"])
            > float(bp["post_handover_rtt_increase_ms"])
        )
        lines.extend(
            (
                "",
                f"Post-event RI detection (no proactive LeoCC stop): "
                f"{'PASS' if reactive_detection else 'REVIEW'}",
                f"Recovery placement between CREO+ and CREO: {'PASS' if placement else 'REVIEW'}",
                f"Additional high-BDP ISL cost: {'PASS' if high_bdp_cost else 'REVIEW'}",
                f"BPL normalized recovery position (CREO+=0, CREO=1): "
                f"{(leo_recovery - plus_recovery) / (creo_recovery - plus_recovery):.3f}",
                "The result is accepted only from the configured 50 ms bidirectional outage, "
                "10 ms RI probes, 45 ms RI threshold, and the published LeoCC adaptation.",
                "The ISL result is reported separately because its longer feedback loop delays "
                "RI observation and increases the high-BDP queue-drain cost.",
                "This is a physical and algorithmic consistency check, not a curve-ranking "
                "parameter search.",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ns3_dir = Path(__file__).resolve().parents[1]
    results_dir = ns3_dir / "results" / "handover"
    if not args.skip_build:
        subprocess.run(
            [str(ns3_dir / "ns3"), "build", "leocc-connected-eval"],
            cwd=ns3_dir,
            check=True,
        )

    path_modes = ("BP", "ISL") if args.case == "all" else (args.case,)
    rows: list[dict[str, object]] = []
    for path_mode in path_modes:
        output_dir = results_dir / f"LeoCC-HO-upload-{path_mode}"
        target_min_rtt_ms = 50.0 if path_mode == "BP" else 100.0
        detection_time = (
            args.handover_time
            + args.handover_duration_ms / 1000.0
            + target_min_rtt_ms / 1000.0
        )
        print(f"Running LeoCC handover {path_mode} -> {output_dir}", flush=True)
        run_ns3(
            ns3_dir,
            {
                "traceSet": "generated",
                "pathMode": path_mode,
                "stopTime": args.duration,
                "throughputPeriod": 0.05,
                "diagnosticPeriod": 0.01,
                "bandwidthJitterStd": 0.10,
                "jitterSeed": args.jitter_seed,
                "errorRate": 0.0,
                "targetMinRtt": target_min_rtt_ms,
                "queuePackets": 500,
                "deviceQueuePackets": 100,
                "outputDir": output_dir.relative_to(ns3_dir),
                "enableReconfiguration": "false",
                "reconfigurationInterval": args.duration + 100.0,
                "reconfigurationOffset": args.duration + 100.0,
                "reconfigurationNotificationTime": detection_time,
                "enableHandover": "true",
                "handoverTime": args.handover_time,
                "handoverDurationMs": args.handover_duration_ms,
            },
            output_dir,
        )
        (output_dir / "phase-config.txt").write_text(
            "phase=handover\n"
            f"path={path_mode}\n"
            f"interruption_start_s={args.handover_time}\n"
            f"interruption_duration_ms={args.handover_duration_ms}\n"
            f"ri_detection_time_s={detection_time:.6f}\n"
            "ri_probe_interval_ms=10\n"
            "ri_outage_threshold_ms=45\n"
            "ri_detection_model=first_post_outage_probe_response_at_outage_end_plus_path_rtt\n"
            "outage_model=bidirectional_terminal_satellite_blackout_with_tx_gate\n"
            "old_satellite_queue_policy=flush_at_start_and_end\n"
            "figure19_throughput=sender_flowmonitor_throughput\n"
            "receiver_goodput=throughput.dat\n"
            "sender_throughput=tx-throughput.dat\n",
            encoding="utf-8",
        )
        rows.append(
            analyze_case(
                f"LeoCC-{path_mode}",
                output_dir,
                "tx-throughput.dat",
                args.handover_time,
                args.handover_duration_ms,
                0.05,
            )
        )

    if args.case in ("BP", "all"):
        references = (
            ("CREO+", results_dir / "CREO+-HO-upload-BP"),
            ("CREO", results_dir / "CREO-HO-upload-BP"),
        )
        for algorithm, reference_dir in references:
            if (reference_dir / "throughput.dat").exists():
                rows.append(
                    analyze_case(
                        algorithm,
                        reference_dir,
                        "throughput.dat",
                        args.handover_time,
                        args.handover_duration_ms,
                        0.05,
                    )
                )

    summary_path = results_dir / "LeoCC-handover-summary.csv"
    csv_rows = [
        {
            key: "" if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    write_summary(summary_path, csv_rows)
    write_acceptance(results_dir / "LeoCC-handover-acceptance.txt", rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
