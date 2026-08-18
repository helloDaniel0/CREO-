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
    parser = argparse.ArgumentParser(description="Evaluate a trained CREO+ model through ns3-ai.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--flows", type=int, default=1)
    parser.add_argument("--sim_seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/creo_single.pt")
    parser.add_argument("--model_dir", type=str, default="models/creo_multi")
    parser.add_argument("--log", type=str, default="creo_test.csv")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--show_ns3_log", action="store_true")
    return parser.parse_args()


def resolve_local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def get_agent(agents, socket_id, args):
    if socket_id in agents:
        return agents[socket_id]
    agent = CreoAgent(eval_mode=True)
    if args.flows == 1:
        agent.load(args.model)
    else:
        per_flow = os.path.join(args.model_dir, f"flow_{socket_id}.pt")
        if os.path.exists(per_flow):
            agent.load(per_flow)
        elif os.path.exists(args.model):
            agent.load(args.model)
    agents[socket_id] = agent
    return agent


def main():
    args = parse_args()
    args.model = resolve_local_path(args.model)
    args.model_dir = resolve_local_path(args.model_dir)
    log_path = resolve_local_path(args.log)
    set_seed(args.seed)
    target = args.target or ("ns3ai_creo_single" if args.flows == 1 else "ns3ai_creo_multi")
    ns3_settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed,
        "flows": args.flows,
    }
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName=target,
        ns3Path=NS3_ROOT,
        ns3Settings=ns3_settings,
        disable_env_checker=True,
    )
    agents = {}
    states = {}

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "socket", "reward", "action_idx", "action_multiplier", "cwnd", "throughput", "rtt_us"])
        try:
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                socket_id = int(obs[0])
                agent = get_agent(agents, socket_id, args)
                state = states.get(socket_id) or agent.observe(obs)
                action_idx, multiplier = agent.choose_action(state, deterministic=True)
                action = np.array([agent.action_to_cwnd(obs, multiplier)], dtype=np.uint64)
                next_obs, _, done, _, _ = env.step(action)
                reward = creo_reward(next_obs)
                states[int(next_obs[0])] = get_agent(agents, int(next_obs[0]), args).observe(next_obs)
                writer.writerow(
                    [
                        step,
                        socket_id,
                        reward,
                        action_idx,
                        multiplier,
                        int(action[0]),
                        float(next_obs[15]),
                        float(next_obs[11]),
                    ]
                )
                obs = next_obs
                step += 1
        except Exception as exc:
            print(f"Exception occurred: {exc}")
            traceback.print_exc()
            return 1
        finally:
            env.close()
    print(f"Saved CREO+ test log to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
