# CTR Predition

## 数据准备
* 将ipinyou数据集放到data/ipinyou目录下
```
info.json - RLB仓库给出的数据集信息，参见https://github.com/han-cai/rlb-dp
train.log.txt - 原始训练集，参见https://github.com/wnzhang/make-ipinyou-data
test.log.txt - 原始测试集，参见https://github.com/wnzhang/make-ipinyou-data
```
* 修改ctr_data.py第270行，指定数据集
* 运行ctr_data.py，为指定数据集生成对应的文件
```
feat.bid.all.txt - 特征映射表
train.bid.all.txt - 特征映射后的训练集
test.bid.all.txt - 特征映射后的测试集
```
## 训练
* 首先修改ctr_main.py第239行指定数据集
* 然后修改config.py第20行指定模型
* 运行ctr_main.py，模型参数会保存在ctr/models/model_params/{数据集}中
## 生成pctr数据
* 修改config.py第20行指定模型
* 修改generate_pctr_data.py第27行指定数据集
* 代码会自动加载最优的模型参数
* 运行generate_pctr_data.py生成pctr数据
```
train.bid.all.hb.csv - HB和DRLB使用的训练集
test.bid.all.hb.csv - HB和DRLB使用的测试集
test.bid.all.rlb.txt - RLB使用的数据集
```