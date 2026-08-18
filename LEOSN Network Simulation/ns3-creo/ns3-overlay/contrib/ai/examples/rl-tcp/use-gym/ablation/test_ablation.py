#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NS3_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../.."))
GYM_BINDING_DIR = os.path.join(NS3_ROOT, "contrib/ai/model/gym-interface/py")
NS3AI_UTILS_DIR = os.path.join(NS3_ROOT, "contrib/ai/python_utils")
for path in (SCRIPT_DIR, GYM_BINDING_DIR, NS3AI_UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import gymnasium as gym
import ns3ai_gym_env  # noqa: F401
import numpy as np

from ablation_agent import VARIANTS, AblationAgent, ablation_reward, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one connected-phase CREO+ ablation.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--sim_seeds", default="7,19,31,43,59")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--mini_window", type=int, default=15)
    parser.add_argument("--bw_trace", default="dataset/SIGCOMMbw.txt")
    parser.add_argument("--latency_trace", default="dataset/SIGCOMMlatency.txt")
    parser.add_argument("--target", default="ns3ai_creo_ablation")
    parser.add_argument("--model", default="")
    parser.add_argument("--raw_log", default="")
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def output_path(path, fallback):
    path = path or fallback
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else math.nan


def mean(values):
    return float(statistics.fmean(values)) if values else math.nan


def transition_recovery_seconds(rows, threshold=0.10, target_fraction=0.90, hold_steps=3):
    recoveries = []
    for index in range(1, len(rows)):
        old_capacity = rows[index - 1]["true_capacity_mbps"]
        new_capacity = rows[index]["true_capacity_mbps"]
        if old_capacity <= 0.0 or abs(new_capacity / old_capacity - 1.0) < threshold:
            continue
        target = target_fraction * new_capacity
        for candidate in range(index, max(index, len(rows) - hold_steps + 1)):
            window = rows[candidate : candidate + hold_steps]
            if len(window) == hold_steps and all(row["throughput_mbps"] >= target for row in window):
                recoveries.append(rows[candidate]["time_s"] - rows[index]["time_s"])
                break
    return mean(recoveries), len(recoveries)


def summarize_seed(variant, sim_seed, rows):
    throughput = [row["throughput_mbps"] for row in rows]
    capacities = [row["true_capacity_mbps"] for row in rows]
    sampled = [row["sampled_capacity_mbps"] for row in rows]
    rtt = [row["rtt_ms"] for row in rows]
    queue = [row["queue_delay_ms"] for row in rows]
    rewards = [row["reward"] for row in rows]
    loss = [row["loss_rate"] for row in rows]
    actions = [row["action_multiplier"] for row in rows]
    feature_time = [row["feature_us"] for row in rows]
    inference_time = [row["inference_us"] for row in rows]
    utilization = [tp / cap for tp, cap in zip(throughput, capacities) if cap > 0.0]
    capacity_errors = [abs(estimate - actual) / actual for estimate, actual in zip(sampled, capacities) if actual > 0.0]
    jitter = [abs(rtt[index] - rtt[index - 1]) for index in range(1, len(rtt))]
    switches = sum(actions[index] != actions[index - 1] for index in range(1, len(actions)))
    recovery, transition_count = transition_recovery_seconds(rows)
    return {
        "variant": variant,
        "sim_seed": sim_seed,
        "samples": len(rows),
        "throughput_mbps": mean(throughput),
        "utilization_pct": 100.0 * mean(utilization),
        "mean_rtt_ms": mean(rtt),
        "p95_rtt_ms": percentile(rtt, 95),
        "mean_queue_delay_ms": mean(queue),
        "p95_queue_delay_ms": percentile(queue, 95),
        "mean_jitter_ms": mean(jitter),
        "p95_jitter_ms": percentile(jitter, 95),
        "loss_pct": 100.0 * mean(loss),
        "capacity_mape_pct": 100.0 * mean(capacity_errors),
        "action_switch_pct": 100.0 * switches / max(len(actions) - 1, 1),
        "transition_recovery_s": recovery,
        "recovered_transitions": transition_count,
        "mean_reward": mean(rewards),
        "mean_feature_us": mean(feature_time),
        "p95_feature_us": percentile(feature_time, 95),
        "mean_inference_us": mean(inference_time),
        "p95_inference_us": percentile(inference_time, 95),
    }


def make_env(args, spec, sim_seed):
    settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": sim_seed,
        "flows": 1,
        "burstPacing": int(spec.use_burst_pacing),
        "miniWindow": args.mini_window,
        "bwTrace": args.bw_trace,
        "latencyTrace": args.latency_trace,
    }
    return gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName=args.target,
        ns3Path=NS3_ROOT,
        ns3Settings=settings,
        disable_env_checker=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    model_path = output_path(args.model, f"models/{args.variant}.pt")
    raw_path = output_path(args.raw_log, f"logs/{args.variant}-test.csv")
    summary_path = output_path(args.summary, f"summaries/{args.variant}.csv")
    agent, _ = AblationAgent.load(model_path, eval_mode=True)
    if agent.spec.name != args.variant:
        raise ValueError(f"model contains {agent.spec.name}, requested {args.variant}")
    sim_seeds = [int(value) for value in args.sim_seeds.split(",") if value.strip()]
    raw_fields = [
        "variant", "sim_seed", "step", "time_s", "reward", "action_idx",
        "action_multiplier", "cwnd_bytes", "throughput_mbps", "sampled_capacity_mbps",
        "true_capacity_mbps", "rtt_ms", "min_rtt_ms", "queue_delay_ms", "loss_rate",
        "feature_us", "inference_us",
    ]
    summaries = []
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(raw_path, "w", newline="", encoding="utf-8") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
        writer.writeheader()
        for sim_seed in sim_seeds:
            env = make_env(args, agent.spec, sim_seed)
            seed_rows = []
            try:
                obs, _ = env.reset()
                agent.builder = type(agent.builder)(
                    agent.spec, agent.capacity_window, agent.history_len
                )
                state = agent.observe(obs)
                done = False
                step = 0
                while not done:
                    inference_start = time.perf_counter_ns()
                    action_idx, multiplier = agent.choose_action(state, deterministic=True)
                    inference_us = (time.perf_counter_ns() - inference_start) / 1000.0
                    cwnd = agent.action_to_cwnd(obs, multiplier)
                    next_obs, _, terminated, truncated, _ = env.step(
                        np.asarray([cwnd], dtype=np.uint64)
                    )
                    done = bool(terminated or truncated)
                    reward = ablation_reward(next_obs)
                    feature_start = time.perf_counter_ns()
                    state = agent.observe(next_obs, multiplier)
                    feature_us = (time.perf_counter_ns() - feature_start) / 1000.0
                    rtt_ms = float(next_obs[11]) / 1000.0
                    min_rtt_ms = float(next_obs[12]) / 1000.0
                    row = {
                        "variant": args.variant,
                        "sim_seed": sim_seed,
                        "step": step,
                        "time_s": float(next_obs[2]) / 1e6,
                        "reward": reward,
                        "action_idx": action_idx,
                        "action_multiplier": multiplier,
                        "cwnd_bytes": cwnd,
                        "throughput_mbps": float(next_obs[15]) / 1e6,
                        "sampled_capacity_mbps": float(next_obs[17]) / 1e6,
                        "true_capacity_mbps": float(next_obs[19]) / 1e6,
                        "rtt_ms": rtt_ms,
                        "min_rtt_ms": min_rtt_ms,
                        "queue_delay_ms": max(rtt_ms - min_rtt_ms, 0.0),
                        "loss_rate": float(next_obs[18]) / 1e6,
                        "feature_us": feature_us,
                        "inference_us": inference_us,
                    }
                    writer.writerow(row)
                    if row["time_s"] >= args.warmup:
                        seed_rows.append(row)
                    obs = next_obs
                    step += 1
                raw_file.flush()
                summary = summarize_seed(args.variant, sim_seed, seed_rows)
                summaries.append(summary)
                print(
                    f"variant={args.variant} seed={sim_seed} "
                    f"util={summary['utilization_pct']:.2f}% "
                    f"rtt={summary['mean_rtt_ms']:.2f}ms"
                )
            finally:
                env.close()

    summary_fields = list(summaries[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"saved raw test log: {raw_path}")
    print(f"saved per-seed summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
