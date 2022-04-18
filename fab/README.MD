# FAB

Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/fab/README-zh.MD)

Take dataset `1458` as an example, run `fab_main.py` to train and test

```
*/RLBid_EA/fab$ python fab_main.py --campaign_id=1458 --time_fraction=96 --budget_para=2 --reward_type=op
```

where `--time_fraction` can take `96` , `48` and `24` , represents different time slot fractions respectively. `--budget_para` can take `2`, `4`, `8` and `16`, represents different budget conditions respectively. `--reward_type` can take `op`, `pctr` , `nn` and `clk`, represents different reward functions in the paper.

```
        clks    real_clks       bids    imps    cost
0 test [257.0, 403.0, 248.20965882453987, 408.9904476322883, 402911, 303274, 22608225.0] 0.38400000000000034
1 test [257.0, 336.0, 248.30861558925923, 341.2193472927829, 358605, 303395, 22608226.0] 0.3130000000000002
……
```

## Results

Bid results of training set and testing set are stored in `fab/result/1458/fab_train_records_op_2.csv` and`fab/result/1458/fab_test_records_op_2.csv` respectively. 

Actions of each time slot on the testing set are stored in `fab/result/1458/fab_test_actions_op_2.csv` respectively. 

