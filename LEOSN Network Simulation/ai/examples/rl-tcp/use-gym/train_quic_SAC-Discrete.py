# Copyright (c) 2020-2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>
#         Muyuan Shen <muyuan_shen@hust.edu.cn>

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from agents import TcpNewRenoAgent, TcpDeepQAgent, TcpQAgent, SACDiscreteAgent
import ns3ai_gym_env
import gymnasium as gym
import sys
import traceback
from collections import deque

STATE_DIM = (8, 10)  # 输入状态向量展平
ACTION_DIM = 7  # 动作空间维度
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99
ALPHA = 0.2  # 熵权重
TAU = 0.005  # 目标网络软更新系数
BUFFER_SIZE = int(1e6)
LSTM_HIDDEN_DIM = 128
initial_bdp = 20

continue_train = True
trained_model_path = "trained model"
existing_model = "sac_discrete_agent_better_action_space.pth"


def save_data_to_file(data, filename):
    with open(filename, 'a') as file:
        row_str = '\t'.join(map(str, data))
        # 写入文件
        file.write(row_str + '\n')


def save_model(agent, filepath, model_name):
    """保存模型及其相关状态"""
    checkpoint = {
        "actor_critic_state_dict": agent.actor_critic.state_dict(),  # Actor-Critic 模型
        "target_actor_critic_state_dict": agent.target_actor_critic.state_dict(),  # Target Actor-Critic 模型
        "optimizer_state_dict": agent.optimizer.state_dict(),  # 优化器状态
        "replay_buffer": agent.replay_buffer,  # 经验回放缓冲区
        "learn_step": agent.learn_step,  # 当前学习步数
        "gamma": agent.gamma,  # 折扣因子
        "tau": agent.tau,  # 软更新系数
        "action_space": agent.action_space,  # 动作空间
    }
    torch.save(checkpoint, filepath + '/' + model_name)
    print(f"Model saved to {filepath}")


def load_model(agent, filepath, model_name):
    """加载模型及其相关状态"""
    checkpoint = torch.load(filepath + '/' + model_name)

    # 加载网络状态
    agent.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    agent.target_actor_critic.load_state_dict(checkpoint["target_actor_critic_state_dict"])

    # 加载优化器状态
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载经验回放缓冲区、超参数和学习步数
    agent.replay_buffer = checkpoint["replay_buffer"]
    agent.learn_step = checkpoint["learn_step"]
    agent.gamma = checkpoint["gamma"]
    agent.tau = checkpoint["tau"]
    agent.action_space = checkpoint["action_space"]

    print(f"Model loaded from {filepath + '/' + model_name}")


def get_agent(socketUuid, useRl):
    #  函数本身可以有属性, 这些属性可以用于存储全局或共享的变量
    agent = get_agent.tcpAgents.get(socketUuid)
    if agent is None:
        if useRl:
            if args.rl_algo == 'SAC':
                agent = SACDiscreteAgent(STATE_DIM, batch_size=BATCH_SIZE)
                if continue_train:
                    load_model(agent, trained_model_path, existing_model)
            elif args.rl_algo == 'DeepQ':
                agent = TcpDeepQAgent()
                print("new Deep Q-learning agent, uuid = {}".format(socketUuid))
            else:
                agent = TcpQAgent()
                print("new Q-learning agent, uuid = {}".format(socketUuid))
        else:
            agent = TcpNewRenoAgent()
            print("new New Reno agent, uuid = {}".format(socketUuid))
        get_agent.tcpAgents[socketUuid] = agent

    return agent


# initialize variable
get_agent.tcpAgents = {}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    help='set seed for reproducibility')
parser.add_argument('--sim_seed', type=int,
                    help='set simulation run number')
parser.add_argument('--duration', type=float,
                    help='set simulation duration (seconds)')
parser.add_argument('--show_log', action='store_true',
                    help='whether show observation and action')
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--result_dir', type=str,
                    default='./rl_tcp_results', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')
parser.add_argument('--rl_algo', type=str,
                    default='DeepQ', help='RL Algorithm, Q or DeepQ')

args = parser.parse_args()
my_seed = 42
if args.seed is not None:
    my_seed = args.seed
print("Python side random seed {}".format(my_seed))
np.random.seed(my_seed)
torch.manual_seed(my_seed)

my_sim_seed = 0
if args.sim_seed:
    my_sim_seed = args.sim_seed

my_duration = 200
if args.duration:
    my_duration = args.duration

if args.use_rl:
    if (args.rl_algo != 'Q') and (args.rl_algo != 'DeepQ') and (args.rl_algo != 'SAC'):
        print("Invalid RL Algorithm {}".format(args.rl_algo))
        exit(1)

res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
            'segmentSize_l', 'bytesInFlight_l']
if args.result:
    for res in res_list:
        globals()[res] = []

stepIdx = 0

action_space = [0.76, 0.95, 0.975, 1, 1.025, 1.05, 1.3]

ns3Settings = {
    'transport_prot': 'QuicRlTimeBased',
    'duration': my_duration,
    'simSeed': my_sim_seed}
env = gym.make("ns3ai_gym_env/Ns3-v0", targetName="ns3ai_rlquic_gym",
               ns3Path="../../../../../", ns3Settings=ns3Settings)
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

throughput_record = deque(maxlen=20)
min_rtt = 100000

try:
    obs, info = env.reset()
    reward = 0
    done = False
    history_obs = deque(maxlen=10)
    # get existing agent or create new TCP agent if needed
    tcpAgent = get_agent(obs[0], args.use_rl)

    while True:
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[9]
        # estimated bytes in flight
        bytesInFlight = obs[7]
        # the throughput in last SP
        throughput = obs[15]
        throughput_record.append(throughput)
        # the rtt in last SP
        rtt = obs[11] / 100
        # min rtt so far
        if 0 < rtt <= min_rtt:
            min_rtt = rtt
        # estimated bdp through cWnd, best throughput and worst rtt
        cwnd_bdp_estimate = obs[16]

        max_throughput = obs[17]

        loss_rate = obs[18]

        # cur_obs = [ssThresh, cWnd, throughput, rtt, min_rtt, bytesInFlight, loss_rate]
        cur_obs = [cWnd / segmentSize, throughput / (1000 * 1000), max(throughput_record),
                   rtt, min_rtt, bytesInFlight, cwnd_bdp_estimate, loss_rate]

        if args.show_log:
            print("Recv obs:",
                  [ssThresh, cWnd / segmentSize, throughput / 10000, max(throughput_record) / 10000, max_throughput,
                   rtt, min_rtt, bytesInFlight, cwnd_bdp_estimate, max(throughput_record) * min_rtt / 10000, loss_rate])

        if args.result:
            for res in res_list:
                globals()[res].append(globals()[res[:-2]])

        if rtt > 0:
            reward = ((1 - loss_rate) * throughput / (rtt * rtt)) / (max(throughput_record) / (min_rtt * min_rtt))
        else:
            reward = 0

        history_obs.append(cur_obs)

        if cwnd_bdp_estimate == 0 or throughput == 0:
            cwnd_bdp_estimate = initial_bdp

        if len(history_obs) < 10:
            init = list(history_obs)
            for _ in range(0, 10 - len(history_obs)):
                init.append(cur_obs)
            action = tcpAgent.get_action(torch.tensor(init, dtype=torch.float32))
        else:
            action = tcpAgent.get_action(torch.tensor(list(history_obs), dtype=torch.float32))

        # data = action
        # data.insert(0,stepIdx+1)

        if args.show_log:
            print("Step:", stepIdx)
            stepIdx += 1
            # save_data_to_file(data, "contrib/ai/examples/rl-tcp/use-gym/dqn_data.txt")
            print(
                f"Normalized Reward (Power): {reward}, Action: {action}, "
                f"Send action: {action * cwnd_bdp_estimate * segmentSize}")

        with open("reward.txt", "a") as f:
            f.write(f"{stepIdx}, {reward}\n")

        # action send to ns-3
        # obs_next, r, done, _, info = env.step(np.array([action * max(throughput_record) * min_rtt / 10000000 *
        #                                                 segmentSize]).astype(np.uint64))
        obs_next, r, done, _, info = env.step(np.array([action * cwnd_bdp_estimate *
                                                        segmentSize]).astype(np.uint64))

        ssThresh = obs_next[4]
        cWnd = obs_next[5]
        segmentSize = obs_next[6]
        segmentsAcked = obs_next[9]
        bytesInFlight = obs_next[7]
        throughput = obs_next[15]
        rtt = obs[11] / 100
        bdp_estimate = obs[16]
        max_throughput = obs[17]
        loss_rate = obs[18]

        # next_obs = [ssThresh, cWnd, throughput, rtt, min_rtt, bytesInFlight, loss_rate]
        next_obs = [cWnd / segmentSize, throughput / (1000 * 1000), max_throughput / (1000 * 1000),
                    rtt, min_rtt, bytesInFlight, bdp_estimate, loss_rate]

        # print("store: ", [cur_obs, action, reward, next_obs, done])
        tcpAgent.store_transition(cur_obs, action_space.index(action), reward, next_obs, done)

        obs = obs_next

        if len(tcpAgent.replay_buffer) >= BATCH_SIZE and stepIdx % 16 == 0:
            loss = tcpAgent.learn()

        if done:
            print("Simulation ended")
            break

        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs[0], args.use_rl)

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

else:
    if args.result:
        if args.result_dir:
            if not os.path.exists(args.result_dir):
                os.mkdir(args.result_dir)
        for res in res_list:
            y = globals()[res]
            x = range(len(y))
            plt.clf()
            plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
            plt.xlabel('Step Number')
            plt.title('Information of {}'.format(res[:-2]))
            plt.savefig('{}.png'.format(os.path.join(args.result_dir, res[:-2])))

finally:
    print("Finally exiting...")
    env.close()

save_model(tcpAgent, trained_model_path, existing_model)
