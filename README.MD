# RL-based bidding strategy

This is a repository of the experiment code supporting the paper **Real-time Bidding Strategy in Display Advertising: An Empirical Analysis**.

Chinese README: [README-zh](https://github.com/hzn666/RLBid_EA/blob/master/README-zh.MD)

## Getting Started

### Data
We use benchmark dataset `iPinYou`. To download and  formalize the iPinYou RTB data, please refers to [make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data).

After the formalization, copy `train.log.txt` and `text.log.txt` of each campaign (e.g. `1458`) to `RLBid_EA/data/ipinyou/1458` for further use.
### CTR Prediction

Before training the bidding strategy, you first need to predict the click-through rate of each ad impression. We provide pre-trained FM model parameters for 4 datasets in `RLBid_EA/ctr/models`. You can use them in conjunction with `RLBid_EA/ctr/generate_pctr.py`.

You are of course able to train a CTR predictor. `RLBid_EA/ctr/model.py`  implemented 9 classical click-through prediction models.

| Model  | Paper                                                        | Link                                                                       |
| ------ | ------------------------------------------------------------ |----------------------------------------------------------------------------|
| LR     | Predicting clicks: estimating the click-through rate for new ads | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/1242572.1242643)            |
| FM     | Factorization machines                                       | [\[paper\]](https://ieeexplore.ieee.org/abstract/document/5694074/)        |
| FFM    | Field-aware factorization machines for CTR prediction        | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/2959100.2959134)            |
| W&D    | Wide & deep learning for recommender systems                 | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/2988450.2988454)            |
| PNN    | Product-based neural networks for user response prediction   | [\[paper\]](https://ieeexplore.ieee.org/abstract/document/7837964/)        |
| DeepFM | DeepFM: a factorization-machine based neural network for CTR prediction | [\[paper\]](https://arxiv.org/abs/1703.04247)                              |
| FNN    | Deep learning over multi-field categorical data              | [\[paper\]](https://link.springer.com/chapter/10.1007/978-3-319-30671-1_4) |
| DCN    | Deep & cross network for ad click predictions                | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3124749.3124754)            |
| AFM    | Attentional factorization machines: Learning the weight of feature interactions via attention networks | [\[paper\]](https://arxiv.org/abs/1708.04617)                              |

For details please refer to [README]() .

## Bidding Strategy

This repository implements 2 static bidding strategies and 3 dynamic bidding strategies based on reinforcement learning.

| Model | Paper                                                        | Link                                                                                                                       |
| ----- | ------------------------------------------------------------ |----------------------------------------------------------------------------------------------------------------------------|
| LIN   | Bid optimizing and inventory scoring in targeted online advertising | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/2339530.2339655)                                                            |
| ORTB  | Optimal real-time bidding for display advertising            | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/2623330.2623633) [\[code\]](https://github.com/wnzhang/rtbarbitrage)        |
| RLB   | Real-time bidding by reinforcement learning in display advertising | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3018661.3018702) [\[code\]](https://github.com/han-cai/rlb-dp)              |
| DRLB  | Budget constrained bidding by model-free reinforcement learning in display advertising | [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3269206.3271748)                                                            |
| FAB   | A dynamic bidding strategy based on model-free reinforcement learning in display advertising | [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9258910/) [\[code\]](https://github.com/JiaXingBinggan/FAB_expr) |

For details please refer to the README in each folder.

## License

This project is licensed under the Apache License 2.0.
## Acknowledgments

*  [@JiaXingBinggan](https://github.com/JiaXingBinggan) contributed the CTR prediction model code for this repository.

