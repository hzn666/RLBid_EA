# CTR Prediction

Take dataset 1458 as an example

Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/ctr/README.MD)

## Data 

By placing the `ipinyou` dataset in the `data/ipinyou` directory, your directory structure should be as follows

```
ipinyou
├── 1458
    ├── train.log.txt - training set
    ├── test.log.txt - testing set
├── 3358
└── ...
```

## Feature Indexing

Run `ctr_data.py` to generate the dataset after Feature Indexing

```
*/RLBid_EA/ctr$ python ctr_data.py --campaign_id=1458
```

```
1458
├── feat.bid.txt - feature index
├── train.bid.txt - training set after feature indexing
├── test.bid.txt - training set after feature indexing
└── ...
```

## Train

Run  `ctr_main.py`  for training

```
*/RLBid_EA/ctr$ python ctr_main.py --campaign_id=1458 --ctr_model=FM
```

Support models: `LR`, `FM`, `FFM`, `W&D`, `DeepFM`, `FNN`, `IPNN`, `OPNN`, `DCN`, `AFM`

Best model parameters are stored in `ctr/models/1458`, e.g. `FMbest.pth`

## Generate pCTR

Run `generate_pctr.py` to generate pCTR data for training bidding strategies

```
*/RLBid_EA/ctr$ python generate_pctr.py --campaign_id=1458 --ctr_model=FM
```

```
1458
├── train.bid.lin.csv - training set for LIN, DRLB and so on 
├── test.bid.lin.csv - testing set for LIN, DRLB and so on 
├── test.bid.rlb.txt - dataset for RLB
└── ...
```

