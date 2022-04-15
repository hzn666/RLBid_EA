import os

from ctr_main import get_model, get_dataset
import config
import pandas as pd
import torch
from tqdm import tqdm


if __name__ == '__main__':
    args = config.init_parser()
    train_data, val_data, test_data, field_nums, feature_nums = get_dataset(args)
    device = torch.device(args.device)

    ctr_model = get_model(args.ctr_model, feature_nums, field_nums, args.latent_dims).to(device)
    pretrain_params = torch.load(
        os.path.join(args.save_param_dir, args.campaign_id, args.ctr_model + 'best.pth'))
    ctr_model.load_state_dict(pretrain_params)

    train_ctrs = []
    test_ctrs = []

    counter = 0
    shape = train_data.shape[0]
    iter = int(shape / 1024)
    for batch in tqdm(range(iter)):
        counter = batch
        _ = ctr_model(
            torch.LongTensor(train_data[batch * 1024:(batch + 1) * 1024, 5:].astype(int)).to(
                args.device)).detach().cpu().numpy()
        train_ctrs.extend(_.flatten().tolist())

    tail = ctr_model(
        torch.LongTensor(train_data[(counter + 1) * 1024:, 5:].astype(int)).to(args.device)).detach().cpu().numpy()
    train_ctrs.extend(tail.flatten().tolist())

    counter = 0
    shape = test_data.shape[0]
    iter = int(shape / 1024)
    for batch in tqdm(range(iter)):
        counter = batch
        _ = ctr_model(
            torch.LongTensor(test_data[batch * 1024:(batch + 1) * 1024, 5:].astype(int)).to(
                args.device)).detach().cpu().numpy()
        test_ctrs.extend(_.flatten().tolist())

    tail = ctr_model(
        torch.LongTensor(test_data[(counter + 1) * 1024:, 5:].astype(int)).to(args.device)).detach().cpu().numpy()
    test_ctrs.extend(tail.flatten().tolist())

    # # click + winning price + hour + timestamp + encode 生成FAB要用的数据
    train_fab = {'clk': train_data[:, 0].tolist(),
                 'pctr': train_ctrs,
                 'market_price': train_data[:, 1].tolist(),
                 '24_time_fraction': train_data[:, 2].tolist(),
                 'minutes': train_data[:, 4].tolist(),
                 }

    test_fab = {'clk': test_data[:, 0].tolist(),
                'pctr': test_ctrs,
                'market_price': test_data[:, 1].tolist(),
                '24_time_fraction': test_data[:, 2].tolist(),
                'minutes': test_data[:, 4].tolist()
                }
    #
    data_path = os.path.join(args.data_path, args.dataset_name, args.campaign_id)
    train_df = pd.DataFrame(data=train_fab)
    test_df = pd.DataFrame(data=test_fab)

    # 生成HB数据和DRLB数据
    train_df['day'] = train_df.minutes.apply(lambda x: int(str(x)[6:8]))
    test_df['day'] = test_df.minutes.apply(lambda x: int(str(x)[6:8]))
    train_df['48_time_fraction'] = train_df['24_time_fraction'] * 2 + (
        train_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 30)))
    test_df['48_time_fraction'] = test_df['24_time_fraction'] * 2 + (
        test_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 30)))
    train_df['96_time_fraction'] = train_df['24_time_fraction'] * 4 + (
        train_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))
    test_df['96_time_fraction'] = test_df['24_time_fraction'] * 4 + (
        test_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))

    train_df.to_csv(os.path.join(data_path, 'train.bid.lin.csv'), index=None)
    test_df.to_csv(os.path.join(data_path, 'test.bid.lin.csv'), index=None)

    # 生成RLB数据
    origin_data = test_df
    rlb_data = origin_data[['clk', 'pctr', 'market_price']]
    fout = open(os.path.join(data_path, 'test.bid.rlb.txt'), 'w')
    for index, row in rlb_data.iterrows():
        fout.write(str(int(row['clk'])) + " " + str(int(row['market_price'])) + " " + str(row['pctr']) + '\n')
    fout.close()
