#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics

from ablation_agent import VARIANTS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS = (
    "throughput_mbps",
    "utilization_pct",
    "mean_rtt_ms",
    "p95_queue_delay_ms",
    "p95_jitter_ms",
    "loss_pct",
    "capacity_mape_pct",
    "action_switch_pct",
    "transition_recovery_s",
    "mean_reward",
    "mean_feature_us",
    "p95_feature_us",
    "mean_inference_us",
    "p95_inference_us",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate all CREO+ connected-phase ablations.")
    parser.add_argument("--input_dir", default="summaries")
    parser.add_argument("--csv", default="summaries/ablation-summary.csv")
    parser.add_argument("--markdown", default="summaries/ablation-summary.md")
    return parser.parse_args()


def local(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def finite(values):
    return [value for value in values if math.isfinite(value)]


def main():
    args = parse_args()
    input_dir = local(args.input_dir)
    rows = []
    for variant in VARIANTS:
        path = os.path.join(input_dir, f"{variant}.csv")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as source:
            seed_rows = list(csv.DictReader(source))
        aggregate = {"variant": variant, "seeds": len(seed_rows)}
        for metric in METRICS:
            values = finite([float(row[metric]) for row in seed_rows])
            aggregate[metric] = statistics.fmean(values) if values else math.nan
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(aggregate)
    if not rows:
        raise RuntimeError(f"No per-variant summaries found under {input_dir}")

    output_csv = local(args.csv)
    output_markdown = local(args.markdown)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_markdown), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Variant", "Throughput (Mb/s)", "Utilization (%)", "Mean RTT (ms)",
        "P95 queue (ms)", "P95 jitter (ms)", "Capacity MAPE (%)", "Recovery (s)",
        "Inference (us)",
    ]
    with open(output_markdown, "w", encoding="utf-8") as output:
        output.write("| " + " | ".join(headers) + " |\n")
        output.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            output.write(
                "| {variant} | {throughput_mbps:.3f} | {utilization_pct:.3f} | "
                "{mean_rtt_ms:.3f} | {p95_queue_delay_ms:.3f} | {p95_jitter_ms:.3f} | "
                "{capacity_mape_pct:.3f} | {transition_recovery_s:.3f} | "
                "{mean_inference_us:.3f} |\n".format(**row)
            )
    print(f"saved aggregate CSV: {output_csv}")
    print(f"saved Markdown table: {output_markdown}")


if __name__ == "__main__":
    main()
