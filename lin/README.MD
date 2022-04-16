# LIN
Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/lin/README-zh.MD)

Take dataset `1458` as an example

Run `lin_main.py` to train and test 

```
*/RLBid_EA/lin$ python lin_main.py --campaign_id=1458 --train_type=normal
```

Where `--train_type` can take `normal` and `reverse` ，which obtain the optimal `base bid` of training set and testing set, respectively. 

## Bid Results

Bid results are stored in `1458/normal` 

```
normal
├── train
    ├── day6_train_best_base_bid.csv - optimal base bid of day6
    ├── ...
    ├── day_train_bid_log.csv - big log of individual day
    ├── train_best_base_bid.csv - optimal base bid of training set
    └── train_bid_log.csv - bid log of training set
└── test
    ├── 2_test_bid_action.csv - bid action of 1/2 budget
    ├── ...
    ├── day_test_bid_log.csv - bid log of individual day
    └── test_bid_log.csv - bid log of training set
```
