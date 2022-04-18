# DRLB

Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/drlb/README-zh.MD)

Take dataset `1458` as an example, run `drlb_main.py` to train and test

```
*/RLBid_EA/drlb$ python drlb_main.py --campaign_id=1458 --time_fraction=96
```

where`--time_fraction` can take `96` , `48` and `24`, represents different time slot fractions respectively. 

```
当前预算条件2
第1轮
训练：真实曝光数2783615, 赢标数1772102, 共获得1780.0个点击, 真实点击数2194.0, 获得pCTR1742.108502579387, 预算106200120.5, 花费92049909.0, CPM51.9439112421294
测试：真实曝光数614638, 赢标数32741, 共获得183.0个点击, 真实点击数515.0, 获得pCTR124.08401337495727, 预算22608227.0, 花费1329711.0, CPM40.61302342628509
……
```

## Results

Bid results of training set and testing set are stored in `drlb/result/1458/test_episode_results_*.csv` and  `drlb/result/1458/train_episode_results_*.csv` respectively.

![](https://latex.codecogs.com/svg.image?\\lambda) of each time slot on training set and testing set are stored in `drlb/result/1458/test_episode_lambdas_*.csv` and `drlb/result/1458/train_episode_lambdas_*.csv` respectively.

Actions of each time slot on training set and testing set are stored in `drlb/result/1458/test_episode_actions_*.csv` and `drlb/result/1458/train_episode_actions_*.csv` respectively.

