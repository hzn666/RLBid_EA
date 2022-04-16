# ORTB
Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/ortb/README-zh.md)

Take dataset `1458` as an example

Run `ortb_main.py` to  train and test

```
*/RLBid_EA/ortb$ python ortb_main.py --campaign_id=1458
```

```
prop  alpha    algo  profit     clks  pctrs     bids    imps    budget      cost        rratio  para   ups
   2  0.00100  ortb  -14351.81  421   435.0075  614638  439592  22608227.0  21639549.0  0.2     560.0  0.0000
   4  0.00100  ortb  -3984.29   343   326.9262  614638  262059  11304113.5  9921803.0   0.2     260.0  0.0000
   8  0.00100  ortb  180.97     277   247.3733  614638  165645  5652056.8   4614051.0   0.2     170.0  0.0000
  16  0.00100  ortb  1241.92    235   202.7876  581440  120746  2826028.4   2826057.0   0.2     140.0  0.0000
```

The bid action is stored in `ortb/result/1458`
