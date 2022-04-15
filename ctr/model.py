import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np


def weight_init(layers):
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.day_data.fill_(1)
            layer.bias.day_data.zero_()
        elif isinstance(layer, nn.Linear):
            n = layer.in_features
            y = 1.0 / np.sqrt(n)
            layer.weight.data.uniform_(-y, y)
            layer.bias.day_data.fill_(0)
            # nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')


# 传统的预测点击率模型
class LR(nn.Module):
    def __init__(self,
                 feature_nums,
                 output_dim=1):
        super(LR, self).__init__()
        self.linear = nn.Embedding(feature_nums, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        out = self.bias + torch.sum(self.linear(x), dim=1)
        pctrs = torch.sigmoid(out)

        return pctrs


class FM(nn.Module):
    def __init__(self,
                 feature_nums,
                 latent_dims,
                 output_dim=1):
        super(FM, self).__init__()
        self.linear = nn.Embedding(feature_nums, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.feature_embedding = nn.Embedding(feature_nums, latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

    def forward(self, x):
        """
        :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
        :return: pctrs
        """
        linear_x = x

        second_x = self.feature_embedding(x)

        square_of_sum = torch.sum(second_x, dim=1) ** 2
        sum_of_square = torch.sum(second_x ** 2, dim=1)

        ix = torch.sum(square_of_sum - sum_of_square, dim=1,
                       keepdim=True)  # 若keepdim值为True,则在输出张量中,除了被操作的dim维度值降为1,其它维度与输入张量input相同。

        out = self.bias + torch.sum(self.linear(linear_x), dim=1) + ix * 0.5
        pctrs = torch.sigmoid(out)

        return pctrs


class FFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(FFM, self).__init__()

        self.field_nums = field_nums

        self.linear = nn.Embedding(feature_nums, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        '''
         FFM 每一个field都有一个关于所有特征的embedding矩阵，例如特征age=14，有一个age对应field的隐向量，
         但是相对于country的field有一个其它的隐向量，以此显示出不同field的区别 
       '''
        self.field_feature_embeddings = nn.ModuleList([
            nn.Embedding(feature_nums, latent_dims) for _ in range(field_nums)
        ])  # 相当于建立一个field_nums * feature_nums * latent_dims的三维矩阵
        for embedding in self.field_feature_embeddings:
            nn.init.xavier_uniform_(embedding.weight.day_data)

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        x_embedding = [self.field_feature_embeddings[i](x) for i in range(self.field_nums)]
        second_x = list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                second_x.append(x_embedding[j][:, i] * x_embedding[i][:, j])
        # 因此预先输入了x，所以这里field的下标就对应了feature的下标，例如下标x_embedding[i][:, j]，假设j=3此时j就已经对应x=[13, 4, 5, 33]中的33
        # 总共有n(n-1)/2种组合方式，n=self.field_nums

        second_x = torch.stack(second_x,
                               dim=1)
        # torch.stack(), torch.cat() https://blog.csdn.net/excellent_sun/article/details/95175823

        out = self.bias + torch.sum(self.linear(x), dim=1) + torch.sum(torch.sum(second_x, dim=1), dim=1, keepdim=True)
        pctrs = torch.sigmoid(out)

        return pctrs


# 基于深度学习的点击率预测模型
class WideAndDeep(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(WideAndDeep, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self.embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims

        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
        :return: pctrs
        """
        embedding_input = self.embedding(x)

        out = self.bias + torch.sum(self.linear(x), dim=1) + self.mlp(
            embedding_input.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)


class InnerPNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(InnerPNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims + self.field_nums * (self.field_nums - 1) // 2
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

        self.row, self.col = list(), list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x)

        inner_product_vectors = torch.sum(torch.mul(embedding_x[:, self.row], embedding_x[:, self.col]), dim=2)

        # 內积之和
        cross_term = inner_product_vectors

        cat_x = torch.cat([embedding_x.view(-1, self.field_nums * self.latent_dims), cross_term], dim=1)

        out = self.mlp(cat_x)

        return torch.sigmoid(out)


class OuterPNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(OuterPNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.latent_dims + self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

        # kernel指的对外积的转换
        self.kernel = torch.ones((self.latent_dims, self.latent_dims)).cuda()
        # nn.init.normal_(self.kernel)

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x)

        sum_embedding_x = torch.sum(embedding_x, dim=1).unsqueeze(1)
        outer_product = torch.mul(torch.mul(sum_embedding_x, self.kernel), sum_embedding_x)

        cross_item = torch.sum(outer_product, dim=1)  # 降维

        cat_x = torch.cat([embedding_x.view(-1, self.field_nums * self.latent_dims), cross_item], dim=1)
        out = self.mlp(cat_x)

        return torch.sigmoid(out)


class DeepFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(DeepFM, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        # nn.init.xavier_normal_(self.linear.weight)
        # self.bias = nn.Parameter(torch.zeros((output_dim,)))

        # FM embedding
        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        # MLP
        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)  # 7141262125646409

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """

        linear_x = x

        second_x = self.feature_embedding(x)

        square_of_sum = torch.sum(second_x, dim=1) ** 2
        sum_of_square = torch.sum(second_x ** 2, dim=1)

        ix = torch.sum(square_of_sum - sum_of_square, dim=1,
                       keepdim=True)  # 若keepdim值为True,则在输出张量中,除了被操作的dim维度值降为1,其它维度与输入张量input相同。

        out = torch.sum(self.linear(linear_x), dim=1) + ix * 0.5 + self.mlp(
            second_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)


class FNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims):
        super(FNN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            # layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        layers.append(nn.Linear(deep_input_dims, 1))

        weight_init(layers)

        self.mlp = nn.Sequential(*layers)

    def load_embedding(self, pretrain_params):
        self.feature_embedding.weight.data.copy_(
            torch.from_numpy(
                np.array(pretrain_params['feature_embedding.weight'].cpu())
            )
        )

    def forward(self, x):
        """
            :param x: Int tensor of size (batch_size, feature_nums, latent_nums)
            :return: pctrs
        """
        embedding_x = self.feature_embedding(x)
        out = self.mlp(embedding_x.view(-1, self.field_nums * self.latent_dims))

        return torch.sigmoid(out)


class DCN(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(DCN, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        deep_input_dims = self.field_nums * self.latent_dims

        deep_net_layers = list()
        neural_nums = [300, 300, 300]
        self.num_neural_layers = 5

        for neural_num in neural_nums:
            deep_net_layers.append(nn.Linear(deep_input_dims, neural_num))
            # deep_net_layers.append(nn.BatchNorm1d(neural_num))
            deep_net_layers.append(nn.ReLU())
            deep_net_layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neural_num

        weight_init(deep_net_layers)

        self.DN = nn.Sequential(*deep_net_layers)

        cross_input_dims = self.field_nums * self.latent_dims
        self.cross_net_w = nn.ModuleList([
            nn.Linear(cross_input_dims, output_dim) for _ in range(self.num_neural_layers)
        ])

        weight_init(self.cross_net_w)

        self.cross_net_b = nn.ParameterList([
            nn.Parameter(torch.zeros((cross_input_dims,))) for _ in range(self.num_neural_layers)
        ])

        self.linear = nn.Linear(neural_nums[-1] + self.field_nums * self.latent_dims, output_dim)
        # nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        embedding_x = self.feature_embedding(x).view(-1, self.field_nums * self.latent_dims)

        cn_x0, cn_x = embedding_x, embedding_x
        for i in range(self.num_neural_layers):
            cn_x_w = self.cross_net_w[i](cn_x)
            cn_x = cn_x0 * cn_x_w + self.cross_net_b[i] + cn_x
        dn_x = self.DN(embedding_x)
        x_stack = torch.cat([cn_x, dn_x], dim=1)

        out = self.linear(x_stack)

        return torch.sigmoid(out)


class AFM(nn.Module):
    def __init__(self,
                 feature_nums,
                 field_nums,
                 latent_dims,
                 output_dim=1):
        super(AFM, self).__init__()
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims

        self.feature_embedding = nn.Embedding(self.feature_nums, self.latent_dims)
        nn.init.xavier_uniform_(self.feature_embedding.weight.data)

        self.row, self.col = list(), list()
        for i in range(self.field_nums - 1):
            for j in range(i + 1, self.field_nums):
                self.row.append(i), self.col.append(j)

        attention_factor = self.latent_dims

        self.attention_net = nn.Linear(self.latent_dims, attention_factor)  # 隐层神经元数量可以变化,不一定为输入的长度
        n = self.attention_net.in_features
        y = 1.0 / np.sqrt(n)
        self.attention_net.weight.data.uniform_(-y, y)
        self.attention_net.bias.day_data.fill_(0)

        self.attention_softmax = nn.Linear(attention_factor, 1)

        self.fc = nn.Linear(self.latent_dims, output_dim)

        self.linear = nn.Embedding(self.feature_nums, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        embedding_x = self.feature_embedding(x)

        inner_product = torch.mul(embedding_x[:, self.row], embedding_x[:, self.col])

        attn_scores = F.relu(self.attention_net(inner_product))
        attn_scores = F.softmax(self.attention_softmax(attn_scores), dim=1)

        attn_scores = F.dropout(attn_scores, p=0.2)
        attn_output = torch.sum(torch.mul(attn_scores, inner_product), dim=1)  # shape: batch_size-latent_dims
        attn_output = F.dropout(attn_output, p=0.2)

        out = self.bias + torch.sum(self.linear(x), dim=1) + self.fc(attn_output)

        return torch.sigmoid(out)