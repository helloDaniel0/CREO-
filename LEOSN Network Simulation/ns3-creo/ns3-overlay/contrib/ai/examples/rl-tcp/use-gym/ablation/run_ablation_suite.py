#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

from ablation_agent import VARIANTS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NS3_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../.."))


def parse_args():
    parser = argparse.ArgumentParser(description="Build, train, test, and summarize all ablations.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--sim_seeds", default="7,19,31,43,59")
    parser.add_argument("--bw_trace", default="dataset/SIGCOMMbw.txt")
    parser.add_argument("--latency_trace", default="dataset/SIGCOMMlatency.txt")
    return parser.parse_args()


def run(command, cwd=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    args = parse_args()
    episodes = 1 if args.smoke else args.episodes
    duration = 3.0 if args.smoke else args.duration
    sim_seeds = "7" if args.smoke else args.sim_seeds
    capacity_window = 32 if args.smoke else 300
    batch_size = 8 if args.smoke else 64

    if not args.skip_build:
        run(["./ns3", "build", "ns3ai_creo_ablation"], cwd=NS3_ROOT)
    for variant in VARIANTS:
        if not args.skip_train:
            run(
                [
                    sys.executable,
                    os.path.join(SCRIPT_DIR, "train_ablation.py"),
                    "--variant", variant,
                    "--episodes", str(episodes),
                    "--duration", str(duration),
                    "--capacity_window", str(capacity_window),
                    "--batch_size", str(batch_size),
                    "--bw_trace", args.bw_trace,
                    "--latency_trace", args.latency_trace,
                ]
            )
        run(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "test_ablation.py"),
                "--variant", variant,
                "--duration", str(duration),
                "--warmup", "0.5" if args.smoke else "5.0",
                "--sim_seeds", sim_seeds,
                "--bw_trace", args.bw_trace,
                "--latency_trace", args.latency_trace,
            ]
        )
    run([sys.executable, os.path.join(SCRIPT_DIR, "summarize_ablation.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())

