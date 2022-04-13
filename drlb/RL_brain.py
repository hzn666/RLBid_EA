import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, feature_numbers, action_nums):
        super(Net, self).__init__()

        deep_input_dims = feature_numbers
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
            nn.Linear(neuron_nums[2], action_nums)
        )

    def forward(self, input):
        actions_value = self.mlp(self.bn_input(input))
        # actions_value = self.mlp(input)

        return actions_value


# 定义DeepQNetwork
class DRLB:
    def __init__(
            self,
            action_space,
            action_numbers,
            feature_numbers,
            learning_rate=0.001,
            reward_decay=1,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=100000,
            batch_size=32,
            device='cuda: 0',
    ):
        self.action_space = action_space  # 动作空间 [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.action_numbers = action_numbers  # 动作数量 7
        self.feature_numbers = feature_numbers  # 状态特征数量 7
        self.lr = learning_rate  # 学习率 1e-3
        self.gamma = reward_decay  # 折扣因子 1
        self.epsilon_max = e_greedy  # epsilon 的最大值 0.9
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数 100
        self.memory_size = memory_size  # 经验池
        self.batch_size = batch_size
        self.epsilon = 0.95
        self.device = device

        if not os.path.exists('result-96'):
            os.mkdir('result-96')

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 设置随机数种子
        # setup_seed(1)

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + 3))  # 状态的特征数*2加上动作和奖励和done

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net = Net(self.feature_numbers, self.action_numbers).to(self.device)

        self.target_net = copy.deepcopy(self.eval_net)
        # 优化器
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), momentum=0.95, weight_decay=1e-5)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

        self.cost_his = []  # 记录所有的cost变化，plot画出

    # 经验池存储，s-state, a-action, r-reward, s_-state_
    def store_transition(self, transition):
        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换
        self.memory_counter += 1

    # 重置epsilon
    def reset_epsilon(self, e_greedy):
        self.epsilon = e_greedy

    def up_learn_step(self):
        self.learn_step_counter += 1

    # 选择动作
    def choose_action(self, state):
        torch.cuda.empty_cache()
        # 统一 state 的 shape, torch.unsqueeze()这个函数主要是对数据维度进行扩充
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)

        random_probability = max(self.epsilon, 0.05)  # 论文的取法
        self.eval_net.eval()
        with torch.no_grad():
            if np.random.uniform() > random_probability:
                # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
                actions_value = self.eval_net.forward(state)
                # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor),按维度dim 返回最大值
                # torch.max(a,1) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一行的行索引）
                action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
                action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
            else:
                index = np.random.randint(0, self.action_numbers)
                action = self.action_space[index]  # 随机选择动作
        self.eval_net.train()

        return action

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)

        self.eval_net.eval()
        with torch.no_grad():
            actions_value = self.eval_net.forward(state)
            action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action

    # 定义DQN的学习过程
    def learn(self):
        # 清除显存缓存
        torch.cuda.empty_cache()

        # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print(('\n目标网络参数已经更新\n'))
        self.learn_step_counter += 1

        # 训练过程
        # 从memory中随机抽取batch_size的数据
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = random.sample(range(self.memory_size), self.batch_size)
        else:
            sample_index = random.sample(range(self.memory_counter), self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # 获取到q_next（target_net产生）以及q_eval（eval_net产生）
        # 如store_transition函数中存储所示，state存储在[0, feature_numbers-1]的位置（即前feature_numbets）
        # state_存储在[feature_numbers+1，memory_size]（即后feature_numbers的位置）
        b_s = torch.FloatTensor(batch_memory[:, :self.feature_numbers]).to(self.device)

        b_a = []
        b_a_origin = batch_memory[:, self.feature_numbers].astype(float)
        for action in b_a_origin:
            b_a.append(self.action_space.index(action))
        b_a = torch.unsqueeze(torch.LongTensor(b_a), 1).to(self.device)
        b_r = torch.FloatTensor(batch_memory[:, self.feature_numbers + 1]).to(self.device)
        b_s_ = torch.FloatTensor(batch_memory[:, self.feature_numbers + 2: 2 * (self.feature_numbers + 1)]).to(
            self.device)

        b_is_done = torch.FloatTensor(1 - batch_memory[:, -1]).view(self.batch_size, 1).to(self.device)

        # q_eval w.r.t the action in experience
        # b_a - 1的原因是，出价动作最高300，而数组的最大index为299
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.target_net.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练

        q_target = b_r.view(self.batch_size, 1) + self.gamma * torch.mul(q_next.max(1)[0].view(self.batch_size,
                                                                                               1), b_is_done)
        # 训练eval_net
        loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def control_epsilon(self, t):
        # 逐渐增加epsilon，增加行为的利用性
        r_epsilon = 1e-2  # 降低速率
        self.epsilon = max(0.95 - r_epsilon * t, 0.05)
