# CTR 预测

以数据集`1458`为例

## 数据准备

将 `ipinyou` 数据集放到 `data/ipinyou` 目录下，你的目录结构应该如下：

```
ipinyou
├── 1458
    ├── train.log.txt - 训练集
    ├── test.log.txt - 测试集
├── 3358
└── ...
```

## 特征索引

运行 `ctr_data.py`，生成特征索引后的数据集

```
*/RLBid_EA/ctr$ python ctr_data.py --campaign_id=1458
```

```
1458
├── feat.bid.txt - 特征索引表
├── train.bid.txt - 特征索引后的训练集
├── test.bid.txt - 特征索引后的测试集
└── ...
```
## 训练
运行 `ctr_main.py` 进行训练

```
*/RLBid_EA/ctr$ python ctr_main.py --campaign_id=1458 --ctr_model=FM
```

支持的模型有`LR`, `FM`, `FFM`, `W&D`, `DeepFM`, `FNN`, `IPNN`, `OPNN`, `DCN`, `AFM`

模型参数会保存在 `ctr/models/1458` 中，如 `FMbest.pth`

## 生成pctr数据
运行 `generate_pctr.py` 生成用于训练出价策略的pCTR数据

```
*/RLBid_EA/ctr$ python generate_pctr.py --campaign_id=1458 --ctr_model=FM
```

```
1458
├── train.bid.lin.csv - LIN和DRLB使用的训练集
├── test.bid.lin.csv - LIN和DRLB使用的测试集
├── test.bid.rlb.txt - RLB使用的数据集
└── ...
```
