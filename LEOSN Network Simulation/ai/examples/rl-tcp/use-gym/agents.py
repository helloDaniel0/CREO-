# Copyright (c) 2023 Huazhong University of Science and Technology
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
# Author: Muyuan Shen <muyuan_shen@hust.edu.cn>


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from torch.distributions import Categorical
import random


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
        )

    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 32
        self.observer_shape = 5
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2 * 5 + 2))  # s, a, r, s'
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.99 ** self.memory_counter:  # choose best
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:  # explore
            action = np.random.randint(0, 4)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape + 1])
        r = torch.Tensor(
            sample[:, self.observer_shape + 1:self.observer_shape + 2])
        s_ = torch.Tensor(sample[:, self.observer_shape + 2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, True)[0].data

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TcpNewRenoAgent:

    def __init__(self):
        self.new_cWnd = 0
        self.new_ssThresh = 0
        pass

    def get_action(self, obs, reward, done, info):
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

        self.new_cWnd = 1
        if cWnd < ssThresh:
            # slow start
            if segmentsAcked >= 1:
                self.new_cWnd = cWnd + segmentSize
        if cWnd >= ssThresh:
            # congestion avoidance
            if segmentsAcked > 0:
                adder = 1.0 * (segmentSize * segmentSize) / cWnd
                adder = int(max(1.0, adder))
                self.new_cWnd = cWnd + adder

        self.new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
        return [self.new_ssThresh, self.new_cWnd]


class TcpDeepQAgent:

    def __init__(self):
        self.dqn = DQN()
        self.new_cWnd = None
        self.new_ssThresh = None
        self.s = None
        self.a = None
        self.r = None
        self.s_ = None  # next state

    def get_action(self, obs, reward, done, info):
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

        # update DQN
        self.s = self.s_
        self.s_ = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
        if self.s is not None:  # not first time
            self.r = segmentsAcked - bytesInFlight - cWnd
            self.dqn.store_transition(self.s, self.a, self.r, self.s_)
            if self.dqn.memory_counter > self.dqn.memory_capacity:
                self.dqn.learn()

        # choose action
        self.a = self.dqn.choose_action(self.s_)
        if self.a & 1:
            self.new_cWnd = cWnd + segmentSize
        else:
            if cWnd > 0:
                self.new_cWnd = cWnd + int(max(1, (segmentSize * segmentSize) / cWnd))
        if self.a < 3:
            self.new_ssThresh = 2 * segmentSize
        else:
            self.new_ssThresh = int(bytesInFlight / 2)

        return [self.new_ssThresh, self.new_cWnd]


class TcpQAgent:

    def discretize(self, metric, minval, maxval):
        metric = max(metric, minval)
        metric = min(metric, maxval)
        return int((metric - minval) * (self.discrete_level - 1) / (maxval - minval))

    def __init__(self):
        self.update_times = 0
        self.learning_rate = None
        self.discount_rate = 0.5
        self.discrete_level = 15
        self.epsilon = 0.1  # exploration rate
        self.state_size = 3
        self.action_size = 1
        self.action_num = 4
        self.actions = np.arange(self.action_num, dtype=int)
        self.q_table = np.zeros((*((self.discrete_level,) * self.state_size), self.action_num))
        # print(self.q_table.shape)
        self.new_cWnd = None
        self.new_ssThresh = None
        self.s = None
        self.a = np.zeros(self.action_size, dtype=int)
        self.r = None
        self.s_ = None  # next state

    def get_action(self, obs, reward, done, info):
        # current ssThreshold
        # ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[9]
        # estimated bytes in flight
        bytesInFlight = obs[7]

        cWnd_d = self.discretize(cWnd, 0., 50000.)
        segmentsAcked_d = self.discretize(segmentsAcked, 0., 64.)
        bytesInFlight_d = self.discretize(bytesInFlight, 0., 1000000.)

        self.s = self.s_
        self.s_ = [cWnd_d, segmentsAcked_d, bytesInFlight_d]
        if self.s:  # not first time
            # update Q-table
            self.learning_rate = 0.3 * (0.995 ** (self.update_times // 10))
            self.r = segmentsAcked - bytesInFlight - cWnd
            self.q_table[tuple(self.s)][tuple(self.a)] = (
                (1 - self.learning_rate) * self.q_table[tuple(self.s)][tuple(self.a)] +
                self.learning_rate * (self.r + self.discount_rate * np.max(self.q_table[tuple(self.s_)]))
            )
            self.update_times += 1

        # epsilon-greedy
        if random.uniform(0, 1) < 0.1:
            self.a[0] = np.random.choice(self.actions)
        else:
            self.a[0] = np.argmax(self.q_table[tuple(self.s_)])

        # map action to cwnd and ssthresh
        if self.a[0] & 1:
            self.new_cWnd = cWnd + segmentSize
        else:
            if cWnd > 0:
                self.new_cWnd = cWnd + int(max(1, (segmentSize * segmentSize) / cWnd))
        if self.a[0] < 3:
            self.new_ssThresh = 2 * segmentSize
        else:
            self.new_ssThresh = int(bytesInFlight / 2)

        return [self.new_ssThresh, self.new_cWnd]


# our proposed actor critic net

class HistoryReplayBuffer:
    def __init__(self, buffer_size, batch_size, history_length):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.history_length = history_length
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        """将经验推入经验池"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """从经验池中随机采样，并返回包含历史信息的批量样本"""
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # 随机选择 `batch_size` 个位置
        for _ in range(self.batch_size):
            idx = random.randint(self.history_length, len(self.buffer) - 1)

            # 获取当前状态及其前面 `history_length` 个时间步的历史状态
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for i in range(idx - self.history_length + 1, idx + 1):
                state, action, reward, next_state, done = self.buffer[i]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)

        # 将状态列表转换为适当的 Tensor 格式
        batch_states = torch.tensor(batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return len(self.buffer)


class CNNLSTMActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, lstm_hidden_size=128, cnn_filters=32, kernel_size=3):
        super(CNNLSTMActorCritic, self).__init__()

        # CNN Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim[0], cnn_filters, kernel_size, padding=1),  # Conv1D over the feature dim
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(2)  # Pool over the time axis
        )

        # LSTM Layer
        cnn_out_dim = cnn_filters * 2  # Based on Conv1D output channels
        self.lstm = nn.LSTM(cnn_out_dim, lstm_hidden_size, batch_first=True)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Output logits for discrete action space
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_size + action_dim, 64),  # State + one-hot action
            nn.ReLU(),
            nn.Linear(64, 1)  # Q-value
        )

    def forward(self, x, action=None):
        # x: (batch_size, seq_len, feature_dim) -> Transpose for Conv1D
        x = x.permute(0, 2, 1)  # (batch_size, feature_dim, seq_len)

        cnn_out = self.cnn(x)  # (batch_size, cnn_filters * 2, seq_len)

        # Prepare input for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch_size, seq_len, cnn_filters * 2)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # Use the last hidden state as the state representation

        # Actor output
        action_logits = self.actor(lstm_out)

        # Critic output
        q_value = None
        if action is not None:
            action_one_hot = F.one_hot(action, num_classes=action_logits.size(-1)).float()
            # print("action_one_hot:", action_one_hot)
            # print("action_one_hot shape:", action_one_hot.shape)
            # print("lstm output:", lstm_out.shape)
            # print(torch.cat([lstm_out, action_one_hot], dim=-1))
            critic_input = torch.cat([lstm_out, action_one_hot], dim=-1)
            q_value = self.critic(critic_input)

        return action_logits, q_value


class SACDiscreteAgent:
    def __init__(self, input_dim, gamma=0.99, tau=0.005, lr=1e-3, batch_size=32, history_length=10):
        self.gamma = gamma
        self.tau = tau
        self.action_space = [0.76, 0.95, 0.975, 1, 1.025, 1.05, 1.3]
        self.action_dim = len(self.action_space)
        self.learn_step = 0

        # Actor-Critic networks
        self.actor_critic = CNNLSTMActorCritic(input_dim, self.action_dim)
        self.target_actor_critic = CNNLSTMActorCritic(input_dim, self.action_dim)
        self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Optimizers
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        # self.alpha_optimizer = optim.Adam([self.alpha], lr=lr)

        self.replay_buffer = HistoryReplayBuffer(20000, batch_size, history_length)

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.actor_critic(state.unsqueeze(0))
            # print("logits:", logits)
            probs = F.softmax(logits, dim=-1)
            # print("probs:", probs)
            action_idx = torch.multinomial(probs, 1).item()  # Sample from policy
            # print("action_idx:", action_idx)
            action = self.action_space[action_idx]
        return action

    def update_target_network(self):
        for target_param, param in zip(self.target_actor_critic.parameters(), self.actor_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, state, action, reward, next_state, done):
        # Forward pass
        # print("action:", action)
        # print("action shape:", action.shape)

        action = action[:, -1]
        reward = reward[:, -1]
        done = done[:, -1]

        # print("action:", action)
        # print("action shape:", action.shape)
        logits, q_value = self.actor_critic(state, action)
        # print("logits shape: ", logits.shape)
        # print("q_value shape: ", q_value.shape)
        with torch.no_grad():
            next_logits, _ = self.target_actor_critic(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            # print("next_probs:", next_probs.shape)
            # print("target_actor_critic(next_state):", self.target_actor_critic(next_state)[0])
            next_q_value = torch.sum(next_probs * self.target_actor_critic(next_state)[0], dim=-1)
            # print("next_q_value: ", next_q_value.shape)
            # print("reward: ", reward.shape)
            q_target = reward + self.gamma * (1 - done) * next_q_value

        # Critic loss
        q_value = q_value.squeeze(-1)
        critic_loss = F.mse_loss(q_value, q_target)

        # Actor loss
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pi = log_probs.gather(1, action.unsqueeze(1))
        entropy = -(probs * log_probs).sum(dim=-1).mean()  # Optional entropy term
        actor_loss = -(log_pi.squeeze(-1) * (q_value.detach() - q_target)).mean()

        return critic_loss, actor_loss - 0.01 * entropy

    def learn(self):
        self.learn_step += 1

        # 如果是目标网络更新步数，进行软更新
        # if self.learn_step % 1000 == 0:  # 假设每1000步更新一次
        #     self.update_target_network()

        state, action, reward, next_state, done = self.replay_buffer.sample()
        # state = torch.tensor(state, dtype=torch.float32)
        # action = torch.tensor(action, dtype=torch.long)
        # reward = torch.tensor(reward, dtype=torch.float32)
        # next_state = torch.tensor(next_state, dtype=torch.float32)
        # done = torch.tensor(done, dtype=torch.float32)
        # print(state.shape)
        # print(action.shape)
        # print(reward.shape)
        # print(next_state.shape)
        # print(done.shape)

        # Compute losses
        critic_loss, actor_loss = self.compute_loss(state, action, reward, next_state, done)

        # Backpropagation
        self.optimizer.zero_grad()
        (critic_loss + actor_loss).backward()
        self.optimizer.step()

        # Soft update target network
        self.update_target_network()

    def store_transition(self, state, action, reward, next_state, done):
        """将经验存储到经验池中"""
        self.replay_buffer.push(state, action, reward, next_state, done)
