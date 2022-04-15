# LIN

以数据集 `1458` 为例，运行 `lin_main.py` 进行LIN的训练和测试

```
*/RLBid_EA/lin$ python lin_main.py --campaign_id=1458 --train_type=normal
```

其中，`--train_type` 可以取值 `normal` 和 `reverse` ，分别得到训练集和测试集的最优 `base bid` 。 

## 出价结果

出价结果保存在 `1458/normal` 中

```
normal
├── train
    ├── day6_train_best_base_bid.csv - 单天的最优base bid
    ├── ...
    ├── day_train_bid_log.csv - 单天出价日志
    ├── train_best_base_bid.csv - 整体的最优base bid
    └── train_bid_log.csv - 整体出价日志
└── test
    ├── 2_test_bid_action.csv - 不同预算条件下的出价动作
    ├── ...
    ├── day_test_bid_log.csv - 单天出价日志
    └── test_bid_log.csv - 整体出价日志
```
