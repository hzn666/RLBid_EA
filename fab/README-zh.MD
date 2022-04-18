# FAB

以数据集 `1458` 为例，运行 `fab_main.py` 进行FAB的训练和测试

```
*/RLBid_EA/fab$ python fab_main.py --campaign_id=1458 --time_fraction=96 --budget_para=2 --reward_type=op
```

其中，`--time_fraction` 可以取值 `96` , `48` 和 `24` ，分别代表不同的的时段划分方式 。`--budget_para` 可以取值`2`, `4`, `8` 和 `16`，分别代表不同的预算条件，如 `2` 代表使用总预算的 `1/2` 。`--reward_type` 可以取值 `op`, `pctr` , `nn` 和 `clk`，分别对应论文中不同的奖励函数。

```
        clks    real_clks       bids    imps    cost
0 test [257.0, 403.0, 248.20965882453987, 408.9904476322883, 402911, 303274, 22608225.0] 0.38400000000000034
1 test [257.0, 336.0, 248.30861558925923, 341.2193472927829, 358605, 303395, 22608226.0] 0.3130000000000002
……
```

## 结果

训练集和测试集的出价结果分别保存在 `fab/result/1458/fab_train_records_op_2.csv` 和`fab/result/1458/fab_test_records_op_2.csv`中。

测试集中每个时段的动作分别保存在 `fab/result/1458/fab_test_actions_nn_2.csv` 中。
