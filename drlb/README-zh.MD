# DRLB

以数据集 `1458` 为例，运行 `drlb_main.py` 进行DRLB的训练和测试

```
*/RLBid_EA/drlb$ python drlb_main.py --campaign_id=1458 --time_fraction=96
```

其中，`--time_fraction` 可以取值 `96` , `48` 和 `24` ，分别代表不同的的时段划分方式 。 

```
当前预算条件2
第1轮
训练：真实曝光数2783615, 赢标数1772102, 共获得1780.0个点击, 真实点击数2194.0, 获得pCTR1742.108502579387, 预算106200120.5, 花费92049909.0, CPM51.9439112421294
测试：真实曝光数614638, 赢标数32741, 共获得183.0个点击, 真实点击数515.0, 获得pCTR124.08401337495727, 预算22608227.0, 花费1329711.0, CPM40.61302342628509
……
```

## 结果

训练集和测试集的出价结果分别保存在 `drlb/result/1458/test_episode_results_*.csv` 和`drlb/result/1458/train_episode_results_*.csv`中。

训练集和测试集中每个时段的![](https://latex.codecogs.com/svg.image?\\lambda)分别保存在 `drlb/result/1458/test_episode_lambdas_*.csv` 和`drlb/result/1458/train_episode_lambdas_*.csv`中。

训练集和测试集中每个时段的动作分别保存在 `drlb/result/1458/test_episode_actions_*.csv` 和`drlb/result/1458/train_episode_actions_*.csv`中。

