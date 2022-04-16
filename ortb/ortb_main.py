import os.path

import pandas as pd
import random
import math
import argparse


def win_ortb(bid, l):
    return bid * 1. / (bid + l)


def get_optimal_l(train_data):
    bid_upper = 300
    bid_num = {}
    for bid in range(0, bid_upper + 1):
        bid_num[bid] = 0
    for cam in train_data:
        for ad in train_data[cam]:
            mp = ad[1]
            bid_num[mp] += 1

    sum = 0
    for bid in bid_num:
        sum += bid_num[bid]

    if sum == 0:
        return 42

    bid_win = {}
    acc = 0.
    for bid in range(0, bid_upper + 1):
        acc += bid_num[bid]
        bid_win[bid] = acc / sum

    ls = range(1, bid_upper)
    min_loss = 9E50
    optimal_l = -1
    for l in ls:
        loss = 0
        for bid in range(0, bid_upper + 1):
            y = bid_win[bid]
            yp = win_ortb(bid, l)
            loss += (y - yp) * (y - yp)
        if loss < min_loss:
            min_loss = loss
            optimal_l = l
    return optimal_l


def init_cam_data_index(cam_data):
    cam_data_index = {}
    for cam in cam_data:
        cam_data_index[cam] = 0
    return cam_data_index


def next_cam_data(cam_data, cam_data_index, cam_data_length, cam):
    data = -1
    if cam_data_index[cam] < cam_data_length[cam]:
        data = cam_data[cam][cam_data_index[cam]]
        cam_data_index[cam] += 1
    return data


def check_data_ran_out(cam_data_index, cam_data_length, cam_vc, volume):
    for cam in cam_data_index:
        if cam_data_index[cam] >= cam_data_length[cam]:
            cam_vc.pop(cam, None)
    sum = 0
    for cam in cam_data_index:
        sum += cam_data_index[cam]
    if sum >= volume:
        return True
    for cam in cam_data_index:
        if cam_data_index[cam] < cam_data_length[cam]:
            return False
    return True  # all the campaign data runs out, but less than the volume


def sample_cam(cam_vc):
    vsum = 0
    for cam in cam_vc:
        vsum += cam_vc[cam]
    s = random.random() * vsum
    sum = 0.
    for cam in cam_vc:
        sum += cam_vc[cam]
        if not sum < s:
            return cam
    # print "###"
    # print cam_vc
    # print s
    # print sum
    # return -1
    return cam_vc.keys()[random.randint(0, len(cam_vc) - 1)]


def bidding(original_ecpc, original_ctr, r, dsp_l, pctr, algo, para):
    return int(math.sqrt(pctr * dsp_l * para / original_ctr + dsp_l * dsp_l) - dsp_l)


def check_lambda_by_profit(cam_data, cam_data_length, cam_r, cam_base_ctr,
                           dsp_budget, volume, dsp_l, cam_v, para, algo):
    # init
    cost = 0
    profit = 0
    budget_run_out = False
    cam_data_index = init_cam_data_index(cam_data)
    cam_vc = cam_v.copy()  # we will change cam_vc when one campaign runs out of data
    data_run_out = check_data_ran_out(cam_data_index, cam_data_length, cam_vc, volume)

    # start simulation
    while (not data_run_out) and (not budget_run_out):
        cam = sample_cam(cam_vc)
        yzp = next_cam_data(cam_data, cam_data_index, cam_data_length, cam)
        if yzp == -1:
            print(cam_data_length)
            print(cam_data_index)
        clk = yzp[0]
        mp = yzp[1]
        pctr = yzp[2]
        r = cam_r[cam]
        bid = bidding(r / cpc_payoff_ratio, cam_base_ctr[cam], r, dsp_l, pctr, algo, para)
        if bid > mp:  # win auction
            cost += mp
            if algo == "lin" or algo == "ortb":
                profit += clk  # these two algorithms care about clicks
            else:
                profit += clk * r - mp * 1.0E-3  # not cpm counting

        budget_run_out = (cost >= dsp_budget)
        data_run_out = check_data_ran_out(cam_data_index, cam_data_length, cam_vc, volume)
    return -profit


def m_step(cam_data, cam_data_length, cam_r, cam_base_ctr, dsp_budget, volume, dsp_l, cam_v, algo_paras, algo):
    optimal_para = -1
    min_loss = 9E100
    paras = algo_paras[algo]
    for para in paras:
        loss = check_lambda_by_profit(cam_data, cam_data_length, cam_r, cam_base_ctr,
                                      dsp_budget, volume, dsp_l, cam_v, para, algo)

        if loss < min_loss:
            min_loss = loss
            optimal_para = para

    return optimal_para


def simulate_one_bidding_strategy_with_parameter(cam_data, cam_data_length, cam_r, cam_original_ecpc, cam_original_ctr,
                                                 dsp_budget, volume, dsp_l, cam_v, algo, para, tag, cm_up_value):
    # init
    cost = 0
    clks = 0
    bids = 0
    bid_list = []
    imps = 0
    win_pctr = 0
    profit = 0
    cam_data_index = init_cam_data_index(cam_data)
    budget_run_out = False
    cam_vc = cam_v.copy()
    data_run_out = check_data_ran_out(cam_data_index, cam_data_length, cam_vc, volume)

    # start simulation
    while (not data_run_out) and (not budget_run_out):
        cam = sample_cam(cam_vc)
        yzp = next_cam_data(cam_data, cam_data_index, cam_data_length, cam)
        original_ecpc = cam_original_ecpc[cam]
        original_ctr = cam_original_ctr[cam]
        clk = yzp[0]
        mp = yzp[1] + cm_up_value  # competition model
        pctr = yzp[2]
        r = cam_r[cam]
        bid = bidding(original_ecpc, original_ctr, r, dsp_l, pctr, algo, para)
        bid_list.append(bid)
        # bid = bidding(original_ecpc, original_ctr, r, config.cam_l[cam], pctr, algo, para)
        bids += 1
        if bid > mp:  # win auction
            imps += 1
            clks += clk
            cost += mp
            win_pctr += pctr
            profit += clk * r - mp * 1.0E-3  # not cpm counting
        budget_run_out = (cost >= dsp_budget)
        data_run_out = check_data_ran_out(cam_data_index, cam_data_length, cam_vc, volume)

    alpha = "uniform"
    if c_alpha != "uniform":
        alpha = "%.5f" % c_alpha

    return "{prop:>4}  {alpha:>7}  {algo:>6}  {profit:>8.2f}  {clks:>4}  {pctrs:>7.4f}  {bids:>7}  {imps:>7}  " \
           "{budget:>10.1f}  {cost:>10.1f}  {rratio:>6.1f}  {para:>6.1f}  {ups:>6.4f}".format(
        prop=tag,
        alpha=alpha,
        algo=algo,
        profit=profit,
        clks=clks,
        pctrs=win_pctr,
        bids=bids,
        imps=imps,
        budget=dsp_budget,
        cost=cost,
        rratio=cpc_payoff_ratio,
        para=para,
        ups=cm_up_value
    ), bid_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign_id', default='1458')
    args = parser.parse_args()

    result_dir = os.path.join('result', args.campaign_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    campaigns = [args.campaign_id]
    bid_algorithms = ["ortb"]
    dsp_l = 42
    budget_proportions = [2, 4, 8, 16]
    cpc_payoff_ratio = 0.2
    c_alpha = 0.001

    algo_one_para = {
        "ortb": 800,
    }
    algo_paras = {
        "ortb": [1 * t for t in range(20, 1400, 30)],
    }
    r = {}  # cpc价值
    v = {args.campaign_id: 1}  # camp选择概率
    train_data = {}
    test_data = {}
    train_data_length = {}
    test_data_length = {}
    ecpc = {}
    base_ctr = {}

    for cam in campaigns:
        train_data[cam] = []
        train_data_df = pd.read_csv("../data/ipinyou/{}/train.bid.lin.csv".format(cam))
        for row in train_data_df.itertuples():
            train_data[cam].append(
                (int(getattr(row, 'clk')), int(getattr(row, 'market_price')), float(getattr(row, 'pctr'))))
        train_data[cam].reverse()
        train_data_length[cam] = len(train_data[cam])
        ecpc[cam] = train_data_df['market_price'].sum() * 1e-3 / train_data_df['clk'].sum()
        r[cam] = ecpc[cam] * cpc_payoff_ratio
        base_ctr[cam] = train_data_df['pctr'].sum() / train_data_length[cam]
        dsp_l = get_optimal_l(train_data)

    test_cost = 0.
    for cam in campaigns:
        test_data[cam] = []
        test_data_df = pd.read_csv("../data/ipinyou/{}/test.bid.lin.csv".format(cam))
        for row in test_data_df.itertuples():
            test_data[cam].append(
                (int(getattr(row, 'clk')), int(getattr(row, 'market_price')), float(getattr(row, 'pctr'))))
            test_cost += int(getattr(row, 'market_price'))
        test_data_length[cam] = len(test_data[cam])

    cam = campaigns[0]
    volume = test_data_length[cam]

    print("prop  alpha      algo  profit     clks  pctrs      bids     imps    budget      cost          rratio  para    ups")

    for proportion in budget_proportions:
        dsp_budget = test_cost / proportion

        for algo in bid_algorithms:
            algo_one_para[algo] = m_step(train_data, train_data_length, r, base_ctr, dsp_budget, volume, dsp_l, v,
                                         algo_paras, algo)

        for algo in bid_algorithms:
            perf, bid_list = simulate_one_bidding_strategy_with_parameter(test_data, test_data_length, r, ecpc,
                                                                          base_ctr,
                                                                          dsp_budget, volume, dsp_l, v.copy(), algo,
                                                                          algo_one_para[algo], "%s" % proportion, 0)
            print(perf)
            df = pd.DataFrame(bid_list).to_csv("result/{}/{}.csv".format(campaigns[0], str(proportion)))
