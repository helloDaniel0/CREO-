#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USE_GYM_DIR = os.path.dirname(SCRIPT_DIR)
NS3_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../.."))
GYM_BINDING_DIR = os.path.join(NS3_ROOT, "contrib/ai/model/gym-interface/py")
NS3AI_UTILS_DIR = os.path.join(NS3_ROOT, "contrib/ai/python_utils")
for path in (SCRIPT_DIR, USE_GYM_DIR, GYM_BINDING_DIR, NS3AI_UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import gymnasium as gym
import ns3ai_gym_env  # noqa: F401
import numpy as np

from ablation_agent import (
    STATIC_ACTION_SPACE,
    VARIANTS,
    AblationAgent,
    ablation_reward,
    get_spec,
    pdpa_action_space,
    read_capacity_trace,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one connected-phase CREO+ ablation.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--sim_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--capacity_window", type=int, default=300)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--mini_window", type=int, default=15)
    parser.add_argument("--pdpa_lag", type=int, default=3)
    parser.add_argument("--pdpa_tolerance", type=float, default=0.025)
    parser.add_argument("--bw_trace", default="dataset/SIGCOMMbw.txt")
    parser.add_argument("--latency_trace", default="dataset/SIGCOMMlatency.txt")
    parser.add_argument("--target", default="ns3ai_creo_ablation")
    parser.add_argument("--model", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--metadata", default="")
    return parser.parse_args()


def ns3_path(path):
    return path if os.path.isabs(path) else os.path.join(NS3_ROOT, path)


def output_path(path, fallback):
    path = path or fallback
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def make_action_space(args, spec):
    if not spec.use_pdpa:
        return list(STATIC_ACTION_SPACE), {
            "mode": "fixed",
            "action_space": list(STATIC_ACTION_SPACE),
        }
    capacities = read_capacity_trace(ns3_path(args.bw_trace))
    actions, metadata = pdpa_action_space(
        capacities,
        lag=args.pdpa_lag,
        tolerance=args.pdpa_tolerance,
    )
    metadata["mode"] = "pdpa"
    metadata["action_space"] = actions
    return actions, metadata


def make_env(args, spec, episode):
    settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed + episode,
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
    if args.episodes < 1 or args.duration <= 0.0:
        raise ValueError("episodes and duration must be positive")
    spec = get_spec(args.variant)
    set_seed(args.seed)
    actions, pdpa_metadata = make_action_space(args, spec)
    model_path = output_path(args.model, f"models/{args.variant}.pt")
    log_path = output_path(args.log, f"logs/{args.variant}-train.csv")
    metadata_path = output_path(args.metadata, f"logs/{args.variant}-config.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    agent = AblationAgent(
        spec=spec,
        action_space=actions,
        capacity_window=args.capacity_window,
        history_len=args.history_len,
        batch_size=args.batch_size,
    )
    fieldnames = [
        "variant",
        "episode",
        "step",
        "time_s",
        "reward",
        "action_idx",
        "action_multiplier",
        "cwnd_bytes",
        "throughput_mbps",
        "sampled_capacity_mbps",
        "true_capacity_mbps",
        "raw_capacity_mbps",
        "denoised_capacity_mbps",
        "rtt_ms",
        "min_rtt_ms",
        "queue_delay_ms",
        "loss_rate",
        "q_loss",
        "actor_loss",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        global_step = 0
        for episode in range(args.episodes):
            env = make_env(args, spec, episode)
            try:
                obs, _ = env.reset()
                state = agent.observe(obs)
                done = False
                episode_reward = 0.0
                episode_step = 0
                while not done:
                    action_idx, multiplier = agent.choose_action(state)
                    cwnd = agent.action_to_cwnd(obs, multiplier)
                    action = np.asarray([cwnd], dtype=np.uint64)
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    reward = ablation_reward(next_obs)
                    next_state = agent.observe(next_obs, multiplier)
                    agent.remember(state, action_idx, reward, next_state, done)
                    losses = agent.learn()
                    q_loss, actor_loss = losses if losses is not None else ("", "")
                    rtt_ms = float(next_obs[11]) / 1000.0
                    min_rtt_ms = float(next_obs[12]) / 1000.0
                    writer.writerow(
                        {
                            "variant": args.variant,
                            "episode": episode,
                            "step": global_step,
                            "time_s": float(next_obs[2]) / 1e6,
                            "reward": reward,
                            "action_idx": action_idx,
                            "action_multiplier": multiplier,
                            "cwnd_bytes": cwnd,
                            "throughput_mbps": float(next_obs[15]) / 1e6,
                            "sampled_capacity_mbps": float(next_obs[17]) / 1e6,
                            "true_capacity_mbps": float(next_obs[19]) / 1e6,
                            "raw_capacity_mbps": agent.builder.last_raw_capacity_mbps,
                            "denoised_capacity_mbps": agent.builder.last_denoised_capacity_mbps,
                            "rtt_ms": rtt_ms,
                            "min_rtt_ms": min_rtt_ms,
                            "queue_delay_ms": max(rtt_ms - min_rtt_ms, 0.0),
                            "loss_rate": float(next_obs[18]) / 1e6,
                            "q_loss": q_loss,
                            "actor_loss": actor_loss,
                        }
                    )
                    state = next_state
                    obs = next_obs
                    episode_reward += reward
                    episode_step += 1
                    global_step += 1
                log_file.flush()
                print(
                    f"variant={args.variant} episode={episode + 1}/{args.episodes} "
                    f"steps={episode_step} reward={episode_reward:.3f}"
                )
            finally:
                env.close()

    run_metadata = {
        "variant": args.variant,
        "spec": spec.__dict__,
        "episodes": args.episodes,
        "duration_s": args.duration,
        "seed": args.seed,
        "sim_seed": args.sim_seed,
        "capacity_window": args.capacity_window,
        "history_len": args.history_len,
        "mini_window": args.mini_window,
        "bw_trace": args.bw_trace,
        "latency_trace": args.latency_trace,
        "pdpa": pdpa_metadata,
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)
    agent.save(model_path, pdpa_metadata)
    print(f"saved model: {model_path}")
    print(f"saved training log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"training failed: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

