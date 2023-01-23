# reference code:https://github.com/megvii-research/ML-GCN
# in action: https://zhuanlan.zhihu.com/p/138107291
# Multi-Label Image Recognition with Graph Convolutional Networks, CVPR 2019.
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy as np
import math
from prettytable import PrettyTable
import torchtext.vocab as vocab
from wama_modules.Encoder import VGGEncoder
from wama_modules.BaseModule import GlobalMaxPool
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


def Cos(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    cos = torch.matmul(x, y.view((-1,))) / (
            (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cos


def gen_A(num_classes, t, adj):
    _adj = adj['adj']
    _nums = adj['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, int)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ML_GCN(nn.Module):
    def __init__(self, encoder, encoder_last_channel, label_category_dict, word_embedding_channel=300, t=0.3, adj=None):
        super().__init__()
        self.encoder = encoder
        self.label_category_dict = label_category_dict
        self.label_name = list(label_category_dict.keys())
        self.num_labels = len(self.label_name)
        self.pooling = GlobalMaxPool()

        self.gc1 = GraphConvolution(word_embedding_channel, 512)
        self.gc2 = GraphConvolution(512, encoder_last_channel)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(self.num_labels, t, adj)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, x, word_embedding):
        # encoder feature extract
        features = self.encoder(x)
        feature = self.pooling(features[-1])

        # 2-layer GCN
        # word embedding
        adj = gen_adj(self.A).detach()
        x = self.gc1(word_embedding, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        logits = torch.matmul(feature, x)

        return_dict = {}
        for label_index, label in enumerate(self.label_name):
            return_dict[label] = logits[:, label_index]

        return return_dict


if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)

    # ML-GCN only support binary labels
    label_category_dict['weather'] = 2  # 5 → 2, force binary classification

    # print dataset
    tb = PrettyTable()
    tb.field_names = ['Case ID']+label_name
    for case_id, case in enumerate(dataset):
        tb.add_row([case_id]+case['label_value'])
    print(tb)

    # get word embedding
    glove = vocab.GloVe(name="6B", dim=300, cache=r'C:\git\.vector_cache')
    total = np.array([])
    for label_index, label in enumerate(label_name):
        if '_' in label:
            _print_str = label +' = '
            _l = list(label.split('_'))  # split with '_'
            _print_str += _l[0]
            emb = glove.vectors[glove.stoi[_l[0]]]
            if len(_l) >1:
                for _l_i in _l[1:]:
                    _print_str += ' + '+_l_i
                    emb += glove.vectors[glove.stoi[_l_i]]
            total = np.append(total, emb.numpy())
            print('label', label_index, _print_str)
        else:
            print('label', label_index, label)
            emb = glove.vectors[glove.stoi[label]]
            total = np.append(total, emb.numpy())
    word_embedding = total.reshape(len(label_name), -1)

    # print similar
    sim_matrx = np.zeros(shape=(len(label_name), len(label_name)))
    for i, li in enumerate(label_name):
        for j, lj in enumerate(label_name):
            if i == j:
                sim_matrx[i, j] = -1
            else:
                sim_matrx[i,j] = Cos(word_embedding[i], word_embedding[j])

    tb = PrettyTable()
    tb.field_names = [' ']+label_name
    for l_index, l in enumerate(label_name):
        tb.add_row([l]+list(sim_matrx[l_index]))
    print('sim_matrx:')
    print(tb)

    # make Co-occurrence matrix and Frequency matrix ------------------
    all_labels = np.array([case['label_value'] for case in dataset])

    # Co-occurrence matrix, shape is (label_num,label_num)
    adj_matrix = np.zeros(shape=(len(label_name), len(label_name)))
    # frequency matrix, shape is (label_num, )
    nums_matrix = np.zeros(shape=(len(label_name)))

    for index in range(all_labels.shape[0]):
        _labels = all_labels[index]
        for i in range(all_labels.shape[1]):
            if _labels[i] == 1:
                nums_matrix[i] += 1
                for j in range(all_labels.shape[1]):
                    if j != i and _labels[j] == 1:
                        adj_matrix[i][j] += 1
    adj = {'adj': adj_matrix, 'nums': nums_matrix}

    # print the two matrixs
    tb = PrettyTable()
    tb.field_names = [' ']+label_name
    for l_index, l in enumerate(label_name):
        tb.add_row([l]+list(adj_matrix[l_index]))
    print('adj_matrix (Co-occurrence matrix):')
    print(tb)

    tb = PrettyTable()
    tb.field_names = [' ']+label_name
    tb.add_row(['total']+list(nums_matrix))
    print('nums_matrix:')
    print(tb)

    # 1D model test------------------------------------------------------
    input = image_1D_tensor
    dim = 1
    print('-' * 22, 'build 1D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    f_channel_list = [64, 128, 256, 512]
    model = ML_GCN(
        encoder=VGGEncoder(
            in_channels=input.shape[1],
            stage_output_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            downsample_ration=[0.8, 0.8, 0.8, 0.8],
            dim=dim),
        encoder_last_channel=f_channel_list[-1],
        label_category_dict=label_category_dict,
        word_embedding_channel=word_embedding.shape[1],
        t=0.3,
        adj=adj)
    pre_logits_dict = model(input, torch.tensor(word_embedding.astype(np.float32)))
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 2D model test------------------------------------------------------
    input = image_2D_tensor
    dim = 2
    print('-' * 22, 'build 2D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    f_channel_list = [64, 128, 256, 512]
    model = ML_GCN(
        encoder=VGGEncoder(
            in_channels=input.shape[1],
            stage_output_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            downsample_ration=[0.8, 0.8, 0.8, 0.8],
            dim=dim),
        encoder_last_channel=f_channel_list[-1],
        label_category_dict=label_category_dict,
        word_embedding_channel=word_embedding.shape[1],
        t=0.3,
        adj=adj)
    pre_logits_dict = model(input, torch.tensor(word_embedding.astype(np.float32)))
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 3D model test------------------------------------------------------
    input = image_3D_tensor
    dim = 3
    print('-' * 22, 'build 3D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    f_channel_list = [64, 128, 256, 512]
    model = ML_GCN(
        encoder=VGGEncoder(
            in_channels=input.shape[1],
            stage_output_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            downsample_ration=[0.5, 0.5, 0.5, 0.8],
            dim=dim),
        encoder_last_channel=f_channel_list[-1],
        label_category_dict=label_category_dict,
        word_embedding_channel=word_embedding.shape[1],
        t=0.3,
        adj=adj)
    pre_logits_dict = model(input, torch.tensor(word_embedding.astype(np.float32)))
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

