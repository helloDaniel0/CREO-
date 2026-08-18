import argparse
import csv
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NS3_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../.."))
GYM_BINDING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../model/gym-interface/py"))
NS3AI_UTILS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../python_utils"))
for path in (GYM_BINDING_DIR, NS3AI_UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import gymnasium as gym
import ns3ai_gym_env
import numpy as np

from creo_agent import CreoAgent, creo_reward, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CREO+ DRL agent on one ns3-ai TCP flow.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sim_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/creo_single.pt")
    parser.add_argument("--log", type=str, default="creo_single_train.csv")
    parser.add_argument("--target", type=str, default="ns3ai_creo_single")
    parser.add_argument("--show_ns3_log", action="store_true")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--capacity_window", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=10)
    return parser.parse_args()


def resolve_local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def main():
    args = parse_args()
    model_path = resolve_local_path(args.model)
    log_path = resolve_local_path(args.log)
    load_path = resolve_local_path(args.load) if args.load else ""
    set_seed(args.seed)

    agent = CreoAgent(
        batch_size=args.batch_size,
        capacity_window=args.capacity_window,
        history_len=args.history_len,
    )
    if load_path:
        agent.load(load_path)

    ns3_settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed,
        "flows": 1,
    }
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName=args.target,
        ns3Path=NS3_ROOT,
        ns3Settings=ns3_settings,
        disable_env_checker=True,
    )

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "socket", "reward", "action_idx", "action_multiplier", "cwnd", "throughput", "rtt_us"])
        try:
            obs, _ = env.reset()
            done = False
            step = 0
            state = agent.observe(obs)

            while not done:
                action_idx, multiplier = agent.choose_action(state)
                action = np.array([agent.action_to_cwnd(obs, multiplier)], dtype=np.uint64)
                next_obs, _, done, _, _ = env.step(action)
                reward = creo_reward(next_obs)
                next_state = agent.observe(next_obs)
                agent.remember(state, action_idx, reward, next_state, done)
                loss = agent.learn()

                writer.writerow(
                    [
                        step,
                        int(obs[0]),
                        reward,
                        action_idx,
                        multiplier,
                        int(action[0]),
                        float(next_obs[15]),
                        float(next_obs[11]),
                    ]
                )
                if loss and step % 50 == 0:
                    print(f"step={step} reward={reward:.4f} loss={loss}")
                state = next_state
                obs = next_obs
                step += 1
        except Exception as exc:
            print(f"Exception occurred: {exc}")
            traceback.print_exc()
            return 1
        finally:
            env.close()

    agent.save(model_path)
    print(f"Saved CREO+ model to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
