import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class Net(nn.Module):
    def __init__(self, feature_num, action_num, reward_num):
        super(Net, self).__init__()

        deep_input_dims = feature_num + action_num
        self.bn_input = nn.BatchNorm1d(deep_input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.fill_(0)

        neuron_nums = [100, 100, 100]
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            # nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], reward_num)
        )

    def forward(self, input):
        actions_value = self.mlp(self.bn_input(input))

        return actions_value


class RewardNet:
    def __init__(self, config):
        self.action_num = 1  # 动作数量 1
        self.reward_num = 1  # 奖励数量 1
        self.feature_num = config['feature_num']  # 动作维度 7
        self.lr = config['lr']  # 学习率
        self.memory_size = config['memory_size']  # 经验池
        self.batch_size = config['batch_size']
        self.device = config['device']

        if not hasattr(self, 'memory_S_counter'):
            self.memory_S_counter = 0

        if not hasattr(self, 'memory_D_counter'):
            self.memory_D_counter = 0

        # 将经验池<状态-动作-累积奖励>中的转换组初始化为0
        self.memory_S = defaultdict()

        # 将经验池<状态-动作-累积奖励中最大>中的转换组初始化为0
        self.memory_D = np.zeros((self.memory_size, self.feature_num + 2))

        self.model_reward, self.real_reward = Net(self.feature_num, self.action_num, self.reward_num).to(
            self.device), Net(self.feature_num, self.action_num, self.reward_num).to(self.device)

        # 优化器
        self.optimizer = torch.optim.RMSprop(self.model_reward.parameters(), momentum=0.95, weight_decay=1e-5)

    def return_model_reward(self, state_action):
        # 统一 observation 的 shape (1, size_of_observation)
        state_action = torch.unsqueeze(torch.FloatTensor(state_action), 0).to(self.device)

        self.model_reward.eval()
        with torch.no_grad():
            model_reward = self.model_reward.forward(state_action).cpu().numpy()

        return model_reward

    def store_S_pair(self, state_action_pair, reward):
        self.memory_S[state_action_pair] = reward

    def get_reward_from_S(self, state_action_pair):
        return self.memory_S.get(state_action_pair, 0)

    def store_D_pair(self, state, action, reward):
        state_action_reward_pair = np.hstack((state, action, reward))

        index = self.memory_D_counter % self.memory_size
        self.memory_D[index, :] = state_action_reward_pair
        self.memory_D_counter += 1

    def learn(self):
        if self.memory_D_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_D_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory_D[sample_index, :]

        state_actions = torch.FloatTensor(batch_memory[:, :self.feature_num + 1]).to(self.device)
        real_reward = torch.unsqueeze(torch.FloatTensor(batch_memory[:, self.feature_num + 1]), 1).to(self.device)

        model_reward = self.model_reward.forward(state_actions)

        loss = F.mse_loss(model_reward, real_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
