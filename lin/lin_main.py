import config
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_budget(df):
    total_budget_list = []

    for day in df.day.unique():
        current_day_budget = np.sum(df[df.day.isin([day])].market_price)
        total_budget_list.append(current_day_budget)

    return total_budget_list


def get_lin_bid(average_pctr, pctr, base_bid):
    bid_price = int((pctr * base_bid) / average_pctr)  # 出价计算
    bid_price = bid_price if bid_price <= 300 else 300  # 限制300
    return bid_price


def bid(average_pctr, data, total_budget_list, budget_prop, base_bid, test_flag=False):
    # 数据统计
    real_imps = len(data)
    budget = np.divide(total_budget_list, budget_prop)

    real_clks = 0  # 真实点击
    win_clks = 0  # 赢得点击数
    win_imps = 0  # 赢标数
    win_pctr = 0  # 赢得pctr
    spend = []
    end_time = []
    day_cpm_list = []
    day_result = []
    bid_action = []

    for day_index, day in enumerate(data.day.unique()):
        current_day_data = data[data.day.isin([day])]
        clks = list(current_day_data['clk'])
        pctrs = list(current_day_data['pctr'])
        market_prices = list(current_day_data['market_price'])
        time_frac = list(current_day_data['24_time_fraction'])
        minutes = list(current_day_data['minutes'])

        day_real_imps = len(current_day_data)
        day_real_clks = 0  # 当天真实点击
        day_win_clks = 0  # 当天赢得点击数
        day_win_imps = 0  # 当天赢标数
        day_win_pctr = 0  # 当天赢得pctr
        day_real_pctr = np.sum(pctrs)
        day_spend = 0  # 花费
        early_stop = False

        today_budget = budget[day_index]

        try:
            with tqdm(range(day_real_imps)) as tqdm_t:
                for impression_index in tqdm_t:
                    day_real_clks += clks[impression_index]
                    bid_price = get_lin_bid(average_pctr, pctrs[impression_index], base_bid)
                    if bid_price >= market_prices[impression_index] and day_spend + market_prices[
                        impression_index] <= today_budget:
                        day_win_pctr += pctrs[impression_index]
                        day_win_imps += 1
                        day_win_clks += clks[impression_index]
                        day_spend += market_prices[impression_index]
                        if test_flag:
                            bid_action.append([pctrs[impression_index], bid_price, market_prices[impression_index],
                                               clks[impression_index], minutes[impression_index], 1])
                    else:
                        if test_flag:
                            bid_action.append([pctrs[impression_index], bid_price, market_prices[impression_index],
                                               clks[impression_index], minutes[impression_index], 0])
                    if not early_stop:
                        if day_spend + market_prices[impression_index] > today_budget:
                            early_stop = time_frac[impression_index]
        except KeyboardInterrupt:
            tqdm_t.close()
            raise
        tqdm_t.close()

        real_clks += day_real_clks
        win_clks += day_win_clks
        win_imps += day_win_imps
        win_pctr += day_win_pctr
        day_cpm = day_spend / day_win_imps
        day_cpm_list.append(day_cpm)
        spend.append(day_spend)

        if not early_stop:
            day_end_time = '{}F'.format(day)
        else:
            day_end_time = str(day) + '_' + str(early_stop)
        end_time.append(day_end_time)

        day_average_pctr = day_real_pctr / day_real_imps
        day_result_temp = [day, day_average_pctr, budget_prop, day_win_clks, day_real_clks,
                           day_win_imps, day_real_imps,
                           day_spend, today_budget, day_cpm, base_bid, day_win_pctr, day_real_pctr, day_end_time]

        day_result.append(day_result_temp)

    cpm = np.sum(spend) / win_imps
    total_result = [average_pctr, budget_prop, win_clks, real_clks, win_imps, real_imps,
                    budget, spend, cpm, day_cpm_list, base_bid, win_pctr, end_time]
    if test_flag:
        return total_result, day_result, bid_action
    else:
        return total_result, day_result


def train(cfg, test_flag=False):
    print('训练类型：', cfg['train_type'])

    train_data = pd.read_csv(cfg['train_data_path'])
    train_data.sort_values(by='minutes', inplace=True)
    total_budget_list = get_budget(train_data)
    base_bid_list = np.arange(1, 301)
    train_result = []
    day_train_result = []

    average_pctr = np.sum(train_data.pctr) / len(train_data)

    for budget_prop in cfg['budget_prop_list']:
        for base_bid in base_bid_list:
            print([camp, budget_prop, base_bid])
            train_result_temp, day_train_result_temp = bid(average_pctr, train_data, total_budget_list, budget_prop,
                                                           base_bid)
            train_result.append(train_result_temp)
            day_train_result.extend(day_train_result_temp)

    print('存储全体训练日志')
    total_train_log = pd.DataFrame(data=train_result,
                                   columns=['average_pctr', 'budget_prop', 'win_clks', 'real_clks', 'win_imps',
                                            'real_imps', 'budget',
                                            'spend', 'cpm', 'day_cpm_list', 'base_bid', 'win_pctr', 'end_time'])
    total_train_log.to_csv(os.path.join(cfg['train_log_path'], 'train_bid_log.csv'), index=False)

    print('存储单天训练日志')
    day_train_log = pd.DataFrame(data=day_train_result,
                                 columns=['day', 'day_average_pctr', 'budget_prop', 'day_win_clks', 'day_real_clks',
                                          'day_win_imps', 'day_real_imps',
                                          'day_spend', 'today_budget', 'day_cpm', 'base_bid', 'day_win_pctr',
                                          'day_real_pctr', 'day_end_time'])
    day_train_log.to_csv(os.path.join(cfg['train_log_path'], 'day_train_bid_log.csv'), index=False)

    print('存储全体训练最优出价')
    best_base_bid = total_train_log.groupby(['budget_prop']).apply(lambda x: x[x.win_clks == x.win_clks.max()])
    best_base_bid.to_csv(os.path.join(cfg['train_log_path'], 'train_best_base_bid.csv'), index=False)

    print('存储单天训练最优出价')
    for day in train_data.day.unique():
        current_day_data = day_train_log[day_train_log.day.isin([day])]
        day_best_base_bid = current_day_data.groupby(['budget_prop']).apply(
            lambda x: x[x.day_win_clks == x.day_win_clks.max()])
        day_best_base_bid.to_csv(os.path.join(cfg['train_log_path'], 'day{}_train_best_base_bid.csv'.format(day)),
                                 index=False)

    if test_flag:
        test(cfg)


def test(cfg):
    print('测试')

    test_data = pd.read_csv(cfg['test_data_path'])
    test_data.sort_values(by='minutes', inplace=True)
    test_result = []
    day_test_result = []
    total_budget_list = get_budget(test_data)
    print(total_budget_list)

    for budget_prop in cfg['budget_prop_list']:
        best_base_bid_df = pd.read_csv(os.path.join(cfg['train_log_path'], 'train_best_base_bid.csv'))
        best_base_bid_row = best_base_bid_df[best_base_bid_df.budget_prop.isin([budget_prop])]
        best_base_bid = best_base_bid_row.iloc[0].base_bid
        avarage_pctr = best_base_bid_row.iloc[0].average_pctr
        print([camp, budget_prop, best_base_bid])
        test_result_temp, day_test_result_temp, bid_action = \
            bid(avarage_pctr, test_data, total_budget_list, budget_prop, best_base_bid, test_flag=True)
        test_result.append(test_result_temp)
        day_test_result.extend(day_test_result_temp)
        print('存储不同预算条件下测试集的出价动作')
        bid_action_df = pd.DataFrame(data=bid_action,
                                     columns=['pctr', 'bid', 'market_price', 'clk', 'minutes', 'purchase'])
        bid_action_df.to_csv(os.path.join(cfg['test_log_path'], '{}_test_bid_action.csv'.format(budget_prop)),
                             index=False)

    print('存储全体测试日志')
    pd.DataFrame(data=test_result,
                 columns=['average_pctr', 'budget_prop', 'win_clks', 'real_clks', 'win_imps',
                          'real_imps',
                          'budget', 'spend', 'cpm', 'day_cpm_list', 'base_bid', 'win_pctr', 'end_time']).to_csv(
        os.path.join(cfg['test_log_path'], 'test_bid_log.csv'), index=False
    )

    print('存储单天测试日志')
    pd.DataFrame(data=day_test_result,
                 columns=['day', 'day_average_pctr', 'budget_prop', 'day_win_clks', 'day_real_clks',
                          'day_win_imps', 'day_real_imps',
                          'day_spend', 'today_budget', 'day_cpm', 'base_bid', 'day_win_pctr',
                          'day_real_pctr', 'day_end_time']).to_csv(
        os.path.join(cfg['test_log_path'], 'day_test_bid_log.csv'), index=False
    )


if __name__ == '__main__':
    config = config.init_parser()
    config = vars(config)

    if config['train_type'] == 'reverse':
        config['train_data'] = 'test.bid.lin.csv'
        config['test_data'] = 'train.bid.lin.csv'
    else:
        config['train_data'] = 'train.bid.lin.csv'
        config['test_data'] = 'test.bid.lin.csv'

    camp = config['campaign_id']

    config['train_data_path'] = os.path.join(config['data'], config['dataset'], camp, config['train_data'])
    config['test_data_path'] = os.path.join(config['data'], config['dataset'], camp, config['test_data'])

    config['result_path'] = os.path.join(config['result'], config['dataset'], camp, config['train_type'])
    config['train_log_path'] = os.path.join(config['result_path'], 'train')
    config['test_log_path'] = os.path.join(config['result_path'], 'test')

    if not os.path.exists(config['train_log_path']):
        os.makedirs(config['train_log_path'])
    if not os.path.exists(config['test_log_path']):
        os.makedirs(config['test_log_path'])

    train(config, test_flag=True)
    # test(config)
