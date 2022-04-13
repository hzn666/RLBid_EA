import torch.utils.data.dataset as Dataset


class libsvm_dataset(Dataset.Dataset):
    def __init__(self, Data, label):
        super(libsvm_dataset, self).__init__()
        # 初始化，定义数据内容和标签
        self.Data = Data
        self.label = label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, item):
        data = self.Data[item]
        label = self.label[item]

        return data, label
