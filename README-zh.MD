# 基于强化学习的出价策略实证分析

本仓库是论文《Real-time Bidding Strategy in Display Advertising: An Empirical Analysis》的代码。

## 开始使用

### 数据
我们使用基准数据集`iPinYou`。数据下载以及标准化请参见 [make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data) 。

数据标准化后，将每个广告活动（如 `1458` ）的 `train.log.txt`  和 `text.log.txt` 文件复制到 `RLBid_EA/data/ipinyou/1458` 以便进一步使用。

### CTR预测

在训练出价策略前，首先需预测每个广告曝光的点击率。我们在 `RLBid_EA/ctr/models` 提供了4个数据集的FM预训练模型参数，你可以配合 `RLBid_EA/ctr/generate_pctr.py` 使用。

你当然也能够自己训练一个CTR预测器。`RLBid_EA/ctr/model.py` 实现了9个经典的点击率预测模型。

| 模型   | 论文                                                         | 链接                                                         |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LR     | Predicting clicks: estimating the click-through rate for new ads | [[paper]](https://dl.acm.org/doi/abs/10.1145/1242572.1242643) |
| FM     | Factorization machines                                       | [[paper]](https://ieeexplore.ieee.org/abstract/document/5694074/) |
| FFM    | Field-aware factorization machines for CTR prediction        | [[paper]](https://dl.acm.org/doi/abs/10.1145/2959100.2959134) |
| W&D    | Wide & deep learning for recommender systems                 | [[paper]](https://dl.acm.org/doi/abs/10.1145/2988450.2988454) |
| PNN    | Product-based neural networks for user response prediction   | [[paper]](https://ieeexplore.ieee.org/abstract/document/7837964/) |
| DeepFM | DeepFM: a factorization-machine based neural network for CTR prediction | [[paper]](https://arxiv.org/abs/1703.04247)                  |
| FNN    | Deep learning over multi-field categorical data              | [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-30671-1_4) |
| DCN    | Deep & cross network for ad click predictions                | [[paper]](https://dl.acm.org/doi/abs/10.1145/3124749.3124754) |
| AFM    | Attentional factorization machines: Learning the weight of feature interactions via attention networks | [[paper]](https://arxiv.org/abs/1708.04617)                  |

具体的使用方法请参见 [README]()

## 出价策略

本仓库实现了2个静态出价策略和3个基于强化学习的动态出价策略。

| 模型 | 论文                                                         | 链接                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LIN  | Bid optimizing and inventory scoring in targeted online advertising | [[paper]](https://dl.acm.org/doi/abs/10.1145/2339530.2339655) |
| ORTB | Optimal real-time bidding for display advertising            | [[paper]](https://dl.acm.org/doi/abs/10.1145/2623330.2623633) [[code]](https://github.com/wnzhang/rtbarbitrage) |
| RLB  | Real-time bidding by reinforcement learning in display advertising | [[paper]](https://dl.acm.org/doi/abs/10.1145/3018661.3018702) [[code]](https://github.com/han-cai/rlb-dp) |
| DRLB | Budget constrained bidding by model-free reinforcement learning in display advertising | [[paper]](https://dl.acm.org/doi/abs/10.1145/3269206.3271748) |
| FAB  | A dynamic bidding strategy based on model-free reinforcement learning in display advertising | [[paper]](https://ieeexplore.ieee.org/abstract/document/9258910/) [[code]](https://github.com/JiaXingBinggan/FAB_expr) |

具体的使用方法请参见每个文件夹下的README

## 许可证

本项目采用 Apache License 2.0

## 致谢

*  [@JiaXingBinggan](https://github.com/JiaXingBinggan) 贡献了本仓库的CTR预测模型代码

