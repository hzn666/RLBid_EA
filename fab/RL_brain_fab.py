import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical, MultivariateNormal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens  # 存储的数据长度
        self.epsilon = 1e-3  # 防止出现zero priority
        self.alpha = 0.6  # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.4  # important sample， 从初始值到1
        self.beta_min = 0.4
        self.beta_max = 1.0
        self.beta_increment_per_sampling = 0.00001
        self.abs_err_upper = 1  # abs_err_upper和epsilon ，表明p优先级值的范围在[epsilon,abs_err_upper]之间，可以控制也可以不控制

        self.memory_size = int(memory_size)
        self.memory_counter = 0

        self.prioritys_ = torch.zeros(size=[self.memory_size, 2]).to(self.device)
        # indexs = torch.range(0, self.memory_size)
        # self.prioritys_[:, 1] = indexs

        self.memory = torch.zeros(size=[self.memory_size, transition_lens]).to(self.device)

    def get_priority(self, td_error):
        return torch.pow(torch.abs(td_error) + self.epsilon, self.alpha)

    def add(self, td_error, transitions):  # td_error是tensor矩阵
        transition_lens = len(transitions)
        p = td_error

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.prioritys_[memory_start: memory_end, :] = p
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.prioritys_[memory_start: self.memory_size, :] = p[0: replace_len_1, :]

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.prioritys_[:replace_len_2, :] = p[replace_len_1: transition_lens, :]

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        if self.memory_counter >= self.memory_size:
            priorities = self.get_priority(self.prioritys_[:, 0:1])

            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            # 采样概率分布
            P = torch.div(priorities, total_p).squeeze(1).cpu().detach().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P, replace=False)).long().to(
                self.device)
        else:
            priorities = self.get_priority(self.prioritys_[:self.memory_counter, 0:1])
            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            P = torch.div(priorities, total_p).squeeze(1).cpu().detach().numpy()
            sample_indexs = torch.Tensor(
                np.random.choice(self.memory_counter, batch_size, p=P, replace=False)).long().to(self.device)

        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()
        # print(self.beta)
        batch = self.memory[sample_indexs]
        choose_priorities = priorities[sample_indexs]
        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

        return sample_indexs, batch, ISweights

    def greedy_sample(self, batch_size):
        # total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.prioritys_)
        else:
            min_prob = torch.min(self.prioritys_[:self.memory_counter, :])
        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        sorted_priorities, sorted_indexs = torch.sort(-self.prioritys_, dim=0)

        choose_idxs = sorted_indexs[:batch_size, :].squeeze(1)

        batch = self.memory[choose_idxs]

        choose_priorities = -sorted_priorities[:batch_size, :]

        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

        return choose_idxs, batch, ISweights

    def batch_update(self, choose_idx, td_errors):
        # p = self.get_priority(td_errors)
        self.prioritys_[choose_idx, 0:1] = td_errors


def weight_init(layers):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            layer.weight.data.uniform_(-0.005, 0.005)
            layer.bias.data.fill_(0)


class Critic(nn.Module):
    def __init__(self, input_dims, action_nums, neuron_nums):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.action_nums = action_nums
        self.neuron_nums = neuron_nums

        deep_input_dims_1 = self.input_dims + self.action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.zero_()

        self.layers_1 = list()
        for neuron_num in self.neuron_nums:
            self.layers_1.append(nn.Linear(deep_input_dims_1, neuron_num))
            # self.layers_1.append(nn.BatchNorm1d(neuron_num))
            self.layers_1.append(nn.ReLU())
            # self.layers_1.append(nn.Dropout(p=0.2))
            deep_input_dims_1 = neuron_num

        self.layers_1.append(nn.Linear(deep_input_dims_1, 1))

        deep_input_dims_2 = self.input_dims + self.action_nums
        self.layers_2 = list()
        for neuron_num in neuron_nums:
            self.layers_2.append(nn.Linear(deep_input_dims_2, neuron_num))
            # self.layers_2.append(nn.BatchNorm1d(neuron_num))
            self.layers_2.append(nn.ReLU())
            # self.layers_2.append(nn.Dropout(p=0.2))
            deep_input_dims_2 = neuron_num
        self.layers_2.append(nn.Linear(deep_input_dims_2, 1))

        self.mlp_1 = nn.Sequential(*self.layers_1)
        self.mlp_2 = nn.Sequential(*self.layers_2)

    def evaluate(self, input, actions):
        obs = self.bn_input(input)
        # obs = input
        c_q_out_1 = self.mlp_1(torch.cat([obs, actions], dim=-1))
        c_q_out_2 = self.mlp_2(torch.cat([obs, actions], dim=-1))

        return c_q_out_1, c_q_out_2

    def evaluate_q_1(self, input, actions):
        obs = self.bn_input(input)
        # obs = input
        c_q_out_1 = self.mlp_1(torch.cat([obs, actions], dim=-1))

        return c_q_out_1


class Actor(nn.Module):
    def __init__(self, input_dims, action_nums, neuron_nums):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_nums
        self.neuron_nums = neuron_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.zero_()

        deep_input_dims = self.input_dims

        self.layers = list()
        for neuron_num in self.neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            # self.layers.append(nn.BatchNorm1d(neuron_num))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, 1))
        weight_init([self.layers[-1]])

        self.layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*self.layers)

    def act(self, input):
        obs = self.bn_input(input)
        # obs = input
        action_means = self.mlp(obs)

        return action_means

    def evaluate(self, input):
        obs = self.bn_input(input)
        # obs = input
        action_means = self.mlp(obs)

        return action_means


class TD3_Model():
    def __init__(
            self,
            neuron_nums,
            input_dims,
            action_nums=1,
            lr_A=3e-4,
            lr_C=3e-4,
            reward_decay=1.0,
            memory_size=100000,
            batch_size=32,
            tau=0.0005,  # for target network soft update
            random_seed=1,
            device='cuda:0',
    ):
        self.action_nums = action_nums
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.memory_size = int(memory_size)
        self.batch_size = batch_size
        self.tau = tau
        self.neuron_nums = neuron_nums
        self.device = device

        setup_seed(random_seed)

        self.memory_counter = 0

        self.input_dims = input_dims

        self.memory = Memory(self.memory_size, self.input_dims * 2 + self.action_nums + 2, self.device)

        self.Actor = Actor(self.input_dims, self.action_nums, self.neuron_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.action_nums, self.neuron_nums).to(self.device)

        self.Actor_ = copy.deepcopy(self.Actor)
        self.Critic_ = copy.deepcopy(self.Critic)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.learn_iter = 0
        self.policy_freq = 3

    def store_transition(self, transitions):  # 所有的值都应该弄成float
        if torch.max(self.memory.prioritys_) == 0.:
            td_errors = torch.cat(
                [torch.ones(size=[len(transitions), 1]).to(self.device), transitions[:, -1].view(-1, 1)],
                dim=-1).detach()
        else:
            td_errors = torch.cat(
                [torch.max(self.memory.prioritys_).expand_as(torch.ones(size=[len(transitions), 1])).to(self.device),
                 transitions[:, -1].view(-1, 1)], dim=-1).detach()
            #
        self.memory.add(td_errors, transitions.detach())

    def choose_action(self, state):
        self.Actor.eval()
        with torch.no_grad():
            actions_means = self.Actor.act(state.to(self.device))
            res_actions = torch.clamp(actions_means + torch.randn_like(actions_means) * 0.1, -0.99, 0.99)
        self.Actor.train()

        return res_actions

    def choose_best_action(self, state):
        self.Actor.eval()
        with torch.no_grad():
            action_means = torch.clamp(self.Actor.evaluate(state), -0.99, 0.99)
        self.Actor.train()

        return action_means

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self):
        self.learn_iter += 1
        self.Actor.train()
        self.Actor_.train()
        self.Critic.train()
        self.Critic_.train()

        # sample
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)
        # if self.memory.memory_counter > self.memory_size:
        #     sample_index = random.sample(range(self.memory_size), self.batch_size)
        # else:
        #     sample_index = random.sample(range(self.memory.memory_counter), self.batch_size)
        #
        # batch_memory = self.memory.memory[sample_index, :]

        b_s = batch_memory[:, :self.input_dims]
        b_a = batch_memory[:, self.input_dims: self.input_dims + self.action_nums]
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_done = torch.unsqueeze(batch_memory[:, -2], 1)
        b_s_ = batch_memory[:, -self.input_dims - 2: -2]  # embedding_layer.forward(batch_memory_states)

        with torch.no_grad():
            actions_means_next = self.Actor_.evaluate(b_s_)
            actions_means_next = torch.clamp(actions_means_next +
                                             torch.clamp(torch.randn_like(actions_means_next) * 0.2, -0.5, 0.5),
                                             -0.99, 0.99)

            q1_target, q2_target = \
                self.Critic_.evaluate(b_s_, actions_means_next)

            q_target = torch.min(q1_target, q2_target)
            q_target = b_r + self.gamma * torch.mul(q_target, 1 - b_done)

        q1, q2 = self.Critic.evaluate(b_s, b_a)

        critic_td_error = (q_target * 2 - q1 - q2).detach() / 2
        critic_loss = (
                ISweights * (
                F.mse_loss(q1, q_target, reduction='none') + F.mse_loss(q2, q_target, reduction='none'))).mean()

        # critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=40, norm_type=2)
        self.optimizer_c.step()

        critic_loss_r = critic_loss.item()
        self.memory.batch_update(choose_idx, critic_td_error)

        if self.learn_iter % self.policy_freq == 0:
            actions_means = self.Actor.evaluate(b_s)
            reg = 0.5 * (torch.pow(actions_means, 2).mean())

            critic_value = self.Critic.evaluate_q_1(b_s, actions_means)
            c_a_loss = -critic_value.mean() + reg * 1e-2

            self.optimizer_a.zero_grad()
            c_a_loss.backward()
            nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=40, norm_type=2)
            self.optimizer_a.step()

            self.soft_update(self.Critic, self.Critic_)
            self.soft_update(self.Actor, self.Actor_)
        return critic_loss_r
