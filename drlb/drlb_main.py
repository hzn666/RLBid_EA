import argparse

import numpy as np
import pandas as pd
import os
from RL_brain import DRLB
from reward_net import RewardNet


def get_budget(train_data):
    _ = []
    for index, day in enumerate(train_data[3].unique()):
        current_day_budget = np.sum(train_data[train_data[3].isin([day])][2])
        _.append(current_day_budget)

    return _


def bid_func(auc_pCTRS, lamda):
    return auc_pCTRS / lamda


def statistics(current_day_budget,
               origin_slot_cost,
               origin_slot_win_imp,
               origin_slot_real_imp,
               origin_slot_win_clk,
               origin_slot_real_clk,
               origin_slot_reward,
               slot_data,
               bid_price_array,
               current_day_imp_num,
               slot):
    # 统计数据

    if current_day_budget[slot] > 0:
        # 如果当前时段的预算有剩余
        if current_day_budget[slot] - origin_slot_cost <= 0 or current_day_imp_num[slot] - origin_slot_real_imp <= 0:
            # 如果当前时段的预算不够买下所有的出价获胜的展示
            temp_slot_real_imp = 0
            temp_slot_cost = 0
            temp_slot_win_imp = 0
            temp_slot_reward = 0
            temp_slot_win_clk = 0
            temp_slot_real_clk = 0

            for i in range(len(slot_data)):
                # 遍历时段的每一条展示
                temp_slot_real_imp += 1
                temp_slot_real_clk += slot_data[i, 0]
                if current_day_imp_num[slot] - temp_slot_real_imp >= 0:
                    if current_day_budget[slot] - temp_slot_cost >= 0:
                        # 如果当前时段的预算能够买下这条展示
                        if slot_data[i, 2] <= bid_price_array[i]:
                            # 如果出价大于了市场价
                            temp_slot_cost += slot_data[i, 2]
                            temp_slot_win_imp += 1
                            temp_slot_win_clk += slot_data[i, 0]
                            temp_slot_reward += slot_data[i, 1]
                    else:
                        # 如果预算已经买不了这条展示，就退出循环
                        break
                else:
                    break
            slot_real_imp = temp_slot_real_imp
            slot_cost = temp_slot_cost if temp_slot_cost > 0 else 0
            slot_win_imp = temp_slot_win_imp
            slot_win_clk = temp_slot_win_clk
            slot_real_clk = temp_slot_real_clk
            slot_reward = temp_slot_reward
        else:
            slot_cost, slot_win_imp, slot_real_imp, slot_win_clk, slot_reward, slot_real_clk = \
                origin_slot_cost, origin_slot_win_imp, origin_slot_real_imp, origin_slot_win_clk, origin_slot_reward, origin_slot_real_clk
    else:
        slot_real_imp = 0
        slot_cost = 0
        slot_win_imp = 0
        slot_win_clk = 0
        slot_real_clk = 0
        slot_reward = 0

    return slot_win_imp, slot_cost, slot_real_imp, slot_reward, slot_win_clk, slot_real_clk


def state_(init_budget,
           slot_data,
           slot_data_pctr,
           slot_lambda,
           action,
           current_day_budget,
           slot,
           current_day_imp_num):
    # 获取当前时段的数据
    slot_data = slot_data.values
    slot_bid_price_array = bid_func(slot_data_pctr, slot_lambda)  # 出价
    slot_bid_price_array = np.where(slot_bid_price_array >= 300, 300, slot_bid_price_array)  # 规范出价
    win_data = slot_data[slot_data[:, 2] <= slot_bid_price_array]  # 筛选赢标的数据

    # 总体统计
    slot_cost = np.sum(win_data[:, 2])  # 当前时段花费
    slot_real_imp = len(slot_data)  # 当前时段真实曝光数
    slot_win_imp = len(win_data)  # 当前时段赢标曝光数
    slot_real_clk = np.sum(slot_data[:, 0])  # 当前时段真实点击数
    slot_win_clk = np.sum(win_data[:, 0])  # 当前时段赢标点击数
    slot_reward = np.sum(win_data[:, 1])  # 当前时段的赢标pctr，按论文中的奖励设置，作为直接奖励
    slot_origin_reward = slot_reward  # 记录原始奖励，和RewardNet得到的奖励区分

    slot_done = 0  # 时段结束标志

    if slot == 0:
        # 第0个时段特殊处理
        if current_day_imp_num[slot] > 0:
            slot_win_imp, slot_cost, slot_real_imp, slot_reward, slot_win_clk, slot_real_clk = \
                statistics(current_day_budget,
                           slot_cost,
                           slot_win_imp,
                           slot_real_imp,
                           slot_win_clk,
                           slot_real_clk,
                           slot_reward,
                           slot_data,
                           slot_bid_price_array,
                           current_day_imp_num,
                           slot)
        else:
            slot_win_imp = 0
            slot_cost = 0
            slot_real_imp = 0
            slot_reward = 0
            slot_win_clk = 0
            slot_real_clk = 0

        # 剩余预算更新
        current_day_budget[slot] = current_day_budget[slot] - slot_cost
        if current_day_budget[slot] < 0:
            current_day_budget[slot] = 0
        # 剩余展示量
        current_day_imp_num[slot] = current_day_imp_num[slot] - slot_real_imp
        if current_day_imp_num[slot] < 0:
            current_day_imp_num[slot] = 0

        # 第0个时段的BCR
        BCR_t_0 = (current_day_budget[slot] - init_budget) / init_budget
        BCR_t = BCR_t_0
    else:
        if current_day_imp_num[slot - 1] > 0:
            slot_win_imp, slot_cost, slot_real_imp, slot_reward, slot_win_clk, slot_real_clk \
                = statistics(current_day_budget,
                             slot_cost,
                             slot_win_imp,
                             slot_real_imp,
                             slot_win_clk,
                             slot_real_clk,
                             slot_reward,
                             slot_data,
                             slot_bid_price_array,
                             current_day_imp_num,
                             slot - 1)
        else:
            slot_real_imp = 0
            slot_cost = 0
            slot_win_imp = 0
            slot_reward = 0
            slot_win_clk = 0
            slot_real_clk = 0

        current_day_budget[slot] = current_day_budget[slot - 1] - slot_cost
        if current_day_budget[slot] < 0:
            slot_done = 1
            current_day_budget[slot] = 0

        current_day_imp_num[slot] = current_day_imp_num[slot - 1] - slot_real_imp
        if current_day_imp_num[slot] < 0:
            slot_done = 1
            current_day_imp_num[slot] = 0

        BCR_t_current = (current_day_budget[slot] - current_day_budget[slot - 1]) / current_day_budget[slot - 1] if \
            current_day_budget[slot - 1] > 0 else 0
        BCR_t = BCR_t_current

    # 状态量定义
    ROL_t = 96 - slot - 1
    CPM_t = slot_cost / slot_win_imp if slot_cost != 0 else 0
    WR_t = slot_win_imp / slot_real_imp if slot_real_imp > 0 else 0
    slot_state = [(slot + 1), current_day_budget[slot] / init_budget, ROL_t, BCR_t, CPM_t, WR_t, slot_origin_reward]

    state_action_t = np.hstack((slot_state, action)).tolist()
    slot_net_reward = Rewardnet.return_model_reward(state_action_t)

    slot_real_clk = np.sum(slot_data[:, 0])

    slot_real_imp = len(slot_data)

    return slot_state, slot_lambda, current_day_budget, slot_net_reward[0][0], slot_origin_reward, slot_win_clk, \
           slot_bid_price_array, current_day_imp_num, slot_win_imp, slot_real_imp, slot_real_clk, slot_cost, slot_reward, slot_done


def choose_init_lamda(config, budget_para):
    base_bid_path = os.path.join('../lin/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before you train DRLB')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_prop'] == budget_para].iloc[0]['base_bid']
    avg_pctr = data[data['budget_prop'] == budget_para].iloc[0]['average_pctr']

    init_lambda = avg_pctr / base_bid

    return init_lambda


def run_train(budget_para):
    # 读取训练数据 并舍弃第一行
    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.bid.lin.csv'))

    header = ['clk', 'pctr', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    train_data = train_data[header]
    # 0:clk, 1:pctr, 2:market_price, 3:day, 4:time_fraction
    train_data.columns = [0, 1, 2, 3, 4]

    # 转换数据格式
    train_data.iloc[:, [0, 2, 3, 4]] = train_data.iloc[:, [0, 2, 3, 4]].astype(int)
    train_data.iloc[:, [1]] = train_data.iloc[:, [1]].astype(float)

    # 计算预算
    total_budget = get_budget(train_data)
    total_budget = np.divide(total_budget, budget_para)

    init_lambda = choose_init_lamda(config, budget_para)

    epoch_train_record = []
    epoch_action = []
    epoch_lambda = []

    optimal_lambda = 0
    epoch_test_record = []
    epoch_test_action = []
    epoch_test_lambda = []

    for epoch in range(config['train_epochs']):
        print('第{}轮'.format(epoch + 1))

        episode_lambda = []
        episode_action = []

        episode_cost = []
        episode_win_imps = []
        episode_real_imps = []
        episode_win_clks = []
        episode_real_clks = []
        episode_win_pctr = []

        RL.reset_epsilon(0.9)

        for day_index, day in enumerate(train_data[3].unique()):
            current_day_data = train_data[train_data[3].isin([day])]

            current_day_budget = [0 for _ in range(config['time_fraction'])]
            current_day_budget[0] = total_budget[day_index]
            current_day_init_budget = total_budget[day_index]

            current_day_imp_num = [0 for _ in range(config['time_fraction'])]
            current_day_imp_num[0] = len(current_day_data)

            current_day_cost = []
            current_day_win_imps = []
            current_day_real_imps = []
            current_day_win_clks = []
            current_day_real_clks = []
            current_day_win_pctr = []

            current_day_state_action_pairs = []

            temp_next_slot_state, temp_next_slot_lambda, temp_next_slot_reward = [], 0, 0

            current_day_action = [-1 for _ in range(config['time_fraction'])]
            current_day_lambda = [0 for _ in range(config['time_fraction'])]
            current_day_lambda[0] = init_lambda

            V = 0  # current episode's cumulative discount reward

            slot_done = 0
            next_slot_action = 0

            for t in range(config['time_fraction']):
                # 更新时段
                slot = t

                # 读取该时段的数据
                # current_slot_data[0] 是否有点击
                # current_slot_data[1] pCTR
                # current_slot_data[2] 市场价格
                # current_slot_data[3] 天
                # current_slot_data[4] 时段
                current_slot_data = current_day_data[current_day_data.iloc[:, 4].isin([slot])]  # 当前时段的数据
                current_slot_data_pctr = current_slot_data.iloc[:, 1].values  # 对应的pctrs

                # 更新RewardNet
                if Rewardnet.memory_D_counter >= config['batch_size']:
                    Rewardnet.learn()

                if t == 0:
                    # 如果是第0个时段（第一个时段）
                    init_action = 0  # 初始动作

                    slot_state, slot_lambda, current_day_budget, slot_net_reward, slot_origin_reward, slot_win_clk, \
                    slot_bid_price_array, current_day_imp_num, slot_win_imp, slot_real_imp, slot_real_clk, slot_cost, \
                    slot_reward, slot_done = state_(current_day_init_budget,
                                                    current_slot_data,
                                                    current_slot_data_pctr,
                                                    init_lambda,
                                                    init_action,
                                                    current_day_budget,
                                                    slot,
                                                    current_day_imp_num)  # 1时段

                    slot_action = RL.choose_action(slot_state)

                    # 第1个时段的数据（第二个时段）
                    next_slot_data = current_day_data[current_day_data.iloc[:, 4].isin([slot + 1])]  # 下一时段的数据
                    next_slot_data_pctr = next_slot_data.iloc[:, 1].values  # 对应的pctrs

                    next_slot_lambda = slot_lambda * (1 + slot_action)
                    next_slot_action = slot_action

                    next_slot_state, next_slot_lambda, current_day_budget, next_slot_net_reward, next_slot_origin_reward, \
                    next_slot_win_clk, next_slot_bid_price_array, current_day_imp_num, next_slot_win_imp, \
                    next_slot_real_imp, next_slot_real_clk, next_slot_cost, next_slot_reward, next_slot_done \
                        = state_(current_day_init_budget,
                                 next_slot_data,
                                 next_slot_data_pctr,
                                 next_slot_lambda,
                                 slot_action,
                                 current_day_budget,
                                 slot + 1,
                                 current_day_imp_num)

                    temp_next_slot_state, temp_next_slot_lambda, current_day_budget, temp_next_slot_reward, current_day_imp_num \
                        = next_slot_state, next_slot_lambda, current_day_budget, next_slot_net_reward, current_day_imp_num
                else:
                    slot_state, slot_lambda, current_day_budget, slot_net_reward, slot_origin_reward, slot_win_clk, \
                    slot_bid_price_array, current_day_imp_num, slot_win_imp, slot_real_imp, slot_real_clk, slot_cost, \
                    slot_reward, slot_done = state_(current_day_init_budget,
                                                    current_slot_data,
                                                    current_slot_data_pctr,
                                                    temp_next_slot_lambda,
                                                    next_slot_action,
                                                    current_day_budget,
                                                    slot,
                                                    current_day_imp_num)

                    slot_action = RL.choose_action(slot_state)

                    next_slot_data = current_day_data[current_day_data.iloc[:, 4].isin([slot + 1])]  # t时段的数据
                    next_slot_data_pctr = next_slot_data.iloc[:, 1].values  # ctrs

                    next_slot_lambda = slot_lambda * (1 + slot_action)
                    next_slot_action = slot_action

                    if t == config['time_fraction'] - 1:
                        slot_done = 1
                        # RL.reset_epsilon(0.05)

                    if t < config['time_fraction'] - 1:
                        next_slot_state, next_slot_lambda, current_day_budget, next_slot_net_reward, next_slot_origin_reward, \
                        next_slot_win_clk, next_slot_bid_price_array, remain_auc_num_next, next_slot_win_imp, \
                        next_slot_real_imp, next_slot_real_clk, next_slot_cost, next_slot_reward, next_slot_done \
                            = state_(current_day_init_budget,
                                     next_slot_data,
                                     next_slot_data_pctr,
                                     next_slot_lambda,
                                     slot_action,
                                     current_day_budget,
                                     slot + 1,
                                     current_day_imp_num)

                        if t + 1 == config['time_fraction'] - 1:
                            optimal_lambda = next_slot_lambda
                            # temp_lambda_record.append(optimal_lambda)
                            # episode_lambda.append(temp_lambda_record)

                    temp_next_slot_state, temp_next_slot_lambda, current_day_budget, temp_next_slot_reward, current_day_imp_num \
                        = next_slot_state, next_slot_lambda, current_day_budget, next_slot_net_reward, current_day_imp_num

                transition = np.hstack((slot_state, slot_action, slot_net_reward, next_slot_state, slot_done))
                RL.store_transition(transition)  # 存储在DRLB的经验池中

                current_day_action[t] = slot_action
                if t < config['time_fraction'] - 1:
                    current_day_lambda[t + 1] = next_slot_lambda

                current_day_cost.append(slot_cost)
                current_day_win_imps.append(slot_win_imp)
                current_day_real_imps.append(slot_real_imp)
                current_day_win_clks.append(slot_win_clk)
                current_day_real_clks.append(slot_real_clk)
                current_day_win_pctr.append(slot_reward)

                current_day_state_action_pairs.append((slot_state, slot_action))
                V += slot_origin_reward

                if RL.memory_counter >= config['batch_size'] and RL.memory_counter % 8 == 0:  # 控制更新速度
                    loss = RL.learn()

                RL.control_epsilon(t + 1)

                if slot_done == 1:
                    break

            # 算法2
            for (s, a) in current_day_state_action_pairs:
                state_action = tuple(np.append(s, a))
                max_rtn = max(Rewardnet.get_reward_from_S(state_action), V)
                Rewardnet.store_S_pair(state_action, max_rtn)
                Rewardnet.store_D_pair(s, a, max_rtn)

            episode_cost.append(np.sum(current_day_cost))
            episode_real_imps.append(np.sum(current_day_real_imps))
            episode_win_imps.append(np.sum(current_day_win_imps))
            episode_win_clks.append(np.sum(current_day_win_clks))
            episode_real_clks.append(np.sum(current_day_real_clks))
            episode_win_pctr.append(np.sum(current_day_win_pctr))
            episode_lambda.append(current_day_lambda)
            episode_action.append(current_day_action)

        print('训练：真实曝光数{}, 赢标数{}, 共获得{}个点击, 真实点击数{}, 获得pCTR{}, 预算{}, 花费{}, CPM{}'
              .format(np.sum(episode_real_imps),
                      np.sum(episode_win_imps),
                      np.sum(episode_win_clks),
                      np.sum(episode_real_clks),
                      np.sum(episode_win_pctr),
                      np.sum(total_budget),
                      np.sum(episode_cost),
                      np.sum(episode_cost) / np.sum(episode_win_imps) if np.sum(episode_win_imps) > 0 else 0))

        test_records, test_actions, test_lambdas = run_test(budget_para)
        epoch_test_record.append(test_records)
        epoch_test_action.append(test_actions)
        epoch_test_lambda.append(test_lambdas)

        episode_result_data = [np.sum(episode_real_imps), np.sum(episode_win_imps), np.sum(episode_win_clks),
                               np.sum(episode_real_clks), np.sum(episode_win_pctr), np.sum(total_budget),
                               np.sum(episode_cost),
                               np.sum(episode_cost) / np.sum(episode_win_imps) if np.sum(episode_win_imps) > 0 else 0]
        epoch_train_record.append(episode_result_data)

        epoch_action.append(episode_action)
        epoch_lambda.append(episode_lambda)

    columns = ['real_imps', 'win_imps', 'clks', 'real_clks', 'pctr', 'budget', 'spent', 'CPM']

    record_path = os.path.join(config['result_path'], config['campaign_id'])
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    test_records_array_df = pd.DataFrame(data=epoch_test_record, columns=columns)
    test_records_array_df.to_csv(record_path + '/test_episode_results_' + str(budget_para) + '.csv')

    test_actions_array_df = pd.DataFrame(data=epoch_test_action)
    test_actions_array_df.to_csv(record_path + '/test_episode_actions_' + str(budget_para) + '.csv')

    test_lambdas_array_df = pd.DataFrame(data=epoch_test_lambda)
    test_lambdas_array_df.to_csv(record_path + '/test_episode_lambdas_' + str(budget_para) + '.csv')

    action_df = pd.DataFrame(data=epoch_action)
    action_df.to_csv(record_path + '/train_episode_actions_' + str(budget_para) + '.csv')

    lambda_df = pd.DataFrame(data=epoch_lambda)
    lambda_df.to_csv(record_path + '/train_episode_lambdas_' + str(budget_para) + '.csv')

    result_data_df = pd.DataFrame(data=epoch_train_record, columns=columns)
    result_data_df.to_csv(record_path + '/train_episode_results_' + str(budget_para) + '.csv')

    return optimal_lambda


def run_test(budget_para):
    # 读取测试数据 并舍弃第一行
    test_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.bid.lin.csv'))

    header = ['clk', 'pctr', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    test_data = test_data[header]

    test_data.columns = [0, 1, 2, 3, 4]

    # 转换数据格式
    test_data.iloc[:, [0, 2, 3, 4]] = test_data.iloc[:, [0, 2, 3, 4]].astype(int)
    test_data.iloc[:, [1]] = test_data.iloc[:, [1]].astype(float)

    # 计算预算
    total_budget = get_budget(test_data)
    total_budget = np.divide(total_budget, budget_para)

    init_lambda = choose_init_lamda(config, budget_para)

    episode_lambda = []
    episode_action = []

    episode_cost = []
    episode_win_imps = []
    episode_real_imps = []
    episode_win_clks = []
    episode_real_clks = []
    episode_win_pctr = []

    for day_index, day in enumerate(test_data[3].unique()):
        current_day_data = test_data[test_data[3].isin([day])]

        current_day_budget = [0 for _ in range(config['time_fraction'])]
        current_day_budget[0] = total_budget[day_index]
        current_day_init_budget = total_budget[day_index]

        current_day_imp_num = [0 for _ in range(config['time_fraction'])]
        current_day_imp_num[0] = len(current_day_data)

        current_day_cost = []
        current_day_win_imps = []
        current_day_real_imps = []
        current_day_win_clks = []
        current_day_real_clks = []
        current_day_win_pctr = []

        temp_next_slot_lambda, temp_B_t_next, temp_remain_t_auctions = 0, [], []

        current_day_action = [-1 for _ in range(config['time_fraction'])]
        current_day_lambda = [0 for _ in range(config['time_fraction'])]
        current_day_lambda[0] = init_lambda

        slot_done = 0
        next_slot_action = 0

        for t in range(config['time_fraction']):
            # 更新时段
            slot = t

            # 读取该时段的数据
            # current_slot_data[0] 是否有点击
            # current_slot_data[1] pCTR
            # current_slot_data[2] 市场价格
            # current_slot_data[3] 天
            # current_slot_data[4] 时段
            current_slot_data = current_day_data[current_day_data.iloc[:, 4].isin([slot])]  # 当前时段的数据
            current_slot_data_pctr = current_slot_data.iloc[:, 1].values  # 对应的pctrs

            if t == 0:
                # 如果是第0个时段（第一个时段）
                init_action = 0

                slot_state, slot_lambda, current_day_budget, slot_net_reward, slot_origin_reward, slot_win_clk, \
                slot_bid_price_array, current_day_imp_num, slot_win_imp, slot_real_imp, slot_real_clk, slot_cost, \
                slot_reward, slot_done = state_(current_day_init_budget,
                                                current_slot_data,
                                                current_slot_data_pctr,
                                                init_lambda,
                                                init_action,
                                                current_day_budget,
                                                slot,
                                                current_day_imp_num)  # 1时段

                slot_action = RL.choose_best_action(slot_state)

                next_slot_lambda = slot_lambda * (1 + slot_action)
                next_slot_action = slot_action
                temp_next_slot_lambda, current_day_budget, current_day_imp_num = next_slot_lambda, current_day_budget, current_day_imp_num
            else:
                slot_state, slot_lambda, current_day_budget, slot_net_reward, slot_origin_reward, slot_win_clk, \
                slot_bid_price_array, current_day_imp_num, slot_win_imp, slot_real_imp, slot_real_clk, slot_cost, \
                slot_reward, slot_done = state_(current_day_init_budget,
                                                current_slot_data,
                                                current_slot_data_pctr,
                                                temp_next_slot_lambda,
                                                next_slot_action,
                                                current_day_budget,
                                                slot,
                                                current_day_imp_num)
                slot_action = RL.choose_best_action(slot_state)

                next_slot_lambda = slot_lambda * (1 + slot_action)
                next_slot_action = slot_action

                if t == config['time_fraction'] - 1:
                    slot_done = 1

                temp_next_slot_lambda = next_slot_lambda

                # if t + 1 == 95:
                #     temp_lambda_record.append(next_slot_lambda)

            current_day_action[t] = slot_action
            if t < config['time_fraction'] - 1:
                current_day_lambda[t + 1] = next_slot_lambda

            current_day_cost.append(slot_cost)
            current_day_win_imps.append(slot_win_imp)
            current_day_real_imps.append(slot_real_imp)
            current_day_win_clks.append(slot_win_clk)
            current_day_real_clks.append(slot_real_clk)
            current_day_win_pctr.append(slot_reward)

            if slot_done == 1:
                break

        episode_cost.append(np.sum(current_day_cost))
        episode_real_imps.append(np.sum(current_day_real_imps))
        episode_win_imps.append(np.sum(current_day_win_imps))
        episode_win_clks.append(np.sum(current_day_win_clks))
        episode_real_clks.append(np.sum(current_day_real_clks))
        episode_win_pctr.append(np.sum(current_day_win_pctr))
        episode_lambda.append(current_day_lambda)
        episode_action.append(current_day_action)

    print('测试：真实曝光数{}, 赢标数{}, 共获得{}个点击, 真实点击数{}, 获得pCTR{}, 预算{}, 花费{}, CPM{}'
          .format(np.sum(episode_real_imps),
                  np.sum(episode_win_imps),
                  np.sum(episode_win_clks),
                  np.sum(episode_real_clks),
                  np.sum(episode_win_pctr),
                  np.sum(total_budget),
                  np.sum(episode_cost),
                  np.sum(episode_cost) / np.sum(episode_win_imps) if np.sum(episode_win_imps) > 0 else 0))

    temp_result = [np.sum(episode_real_imps), np.sum(episode_win_imps), np.sum(episode_win_clks),
                   np.sum(episode_real_clks), np.sum(episode_win_pctr), np.sum(total_budget),
                   np.sum(episode_cost),
                   np.sum(episode_cost) / np.sum(episode_win_imps) if np.sum(episode_win_imps) > 0 else 0]

    return temp_result, episode_action, episode_lambda


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--e_greedy', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--feature_num', type=int, default=7)
    parser.add_argument('--action_num', type=int, default=7)
    parser.add_argument('--budget_para', default=[2, 4, 8, 16])
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--replace_target_iter', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    config = vars(args)

    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])

    budget_para_list = config['budget_para']
    for i in range(len(budget_para_list)):
        # 初始化DRLB
        RL = DRLB([-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08], config)

        # 初始化RN
        Rewardnet = RewardNet(config)

        print('当前预算条件{}'.format(budget_para_list[i]))
        optimal_lambda = run_train(budget_para_list[i])
