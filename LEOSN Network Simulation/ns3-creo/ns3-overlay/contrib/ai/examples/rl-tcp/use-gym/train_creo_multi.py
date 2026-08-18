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
    parser = argparse.ArgumentParser(description="Train CREO+ with multiple competing ns3-ai TCP flows.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--flows", type=int, default=3)
    parser.add_argument("--sim_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models/creo_multi")
    parser.add_argument("--log", type=str, default="creo_multi_train.csv")
    parser.add_argument("--target", type=str, default="ns3ai_creo_multi")
    parser.add_argument("--show_ns3_log", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def resolve_local_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def get_agent(agents, socket_id, args):
    if socket_id not in agents:
        agents[socket_id] = CreoAgent(batch_size=args.batch_size)
        model = os.path.join(args.model_dir, f"flow_{socket_id}.pt")
        if os.path.exists(model):
            agents[socket_id].load(model)
    return agents[socket_id]


def main():
    args = parse_args()
    args.model_dir = resolve_local_path(args.model_dir)
    log_path = resolve_local_path(args.log)
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    ns3_settings = {
        "transport_prot": "TcpRlTimeBased",
        "duration": args.duration,
        "simSeed": args.sim_seed,
        "flows": args.flows,
    }
    env = gym.make(
        "ns3ai_gym_env/Ns3-v0",
        targetName=args.target,
        ns3Path=NS3_ROOT,
        ns3Settings=ns3_settings,
        disable_env_checker=True,
    )

    agents = {}
    states = {}
    last_obs = {}

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
                state = states.get(socket_id)
                if state is None:
                    state = agent.observe(obs)

                action_idx, multiplier = agent.choose_action(state)
                action = np.array([agent.action_to_cwnd(obs, multiplier)], dtype=np.uint64)
                next_obs, _, done, _, _ = env.step(action)
                reward = creo_reward(next_obs)
                next_socket = int(next_obs[0])

                next_agent = get_agent(agents, next_socket, args)
                next_state = next_agent.observe(next_obs)
                agent.remember(state, action_idx, reward, next_state, done)
                loss = agent.learn()

                states[socket_id] = next_state if next_socket == socket_id else state
                states[next_socket] = next_state
                last_obs[socket_id] = obs

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
                if loss and step % 100 == 0:
                    print(f"step={step} flow={socket_id} reward={reward:.4f} loss={loss}")
                obs = next_obs
                step += 1
        except Exception as exc:
            print(f"Exception occurred: {exc}")
            traceback.print_exc()
            return 1
        finally:
            env.close()

    for socket_id, agent in agents.items():
        agent.save(os.path.join(args.model_dir, f"flow_{socket_id}.pt"))
    print(f"Saved {len(agents)} CREO+ flow models to {args.model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
