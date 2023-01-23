# ref code(official):https://github.com/HCPLab-SYSU/SSGRL
# different from ML-GCN，SSGRL support multi-class label

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from wama_modules.Encoder import VGGEncoder
from prettytable import PrettyTable
import torchtext.vocab as vocab
from wama_modules.Head import ClassificationHead
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


def Cos(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    cos = torch.matmul(x, y.view((-1,))) / (
            (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cos


# network structure
class GGNN(nn.Module):
    def __init__(self, input_dim, time_step, in_matrix, out_matrix):
        super(GGNN, self).__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        self._in_matrix = in_matrix
        self._out_matrix = out_matrix

        self.fc_eq3_w = nn.Linear(2 * input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2 * input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2 * input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        batch_size = input.size()[0]
        input = input.view(-1, self.input_dim)
        node_num = self._in_matrix.size()[0]
        batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        batch_in_matrix = self._in_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
        batch_out_matrix = self._out_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
        for t in range(self.time_step):
            # eq(2)
            av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(batch_out_matrix, batch_aog_nodes)),
                           2)
            av = av.view(batch_size * node_num, -1)

            flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)

            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes))

            flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv * hv
            batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
        return batch_aog_nodes


class semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, batch_size, img_feature_map, word_embedding):
        """
        # old implement for only 2D image input

        convsize = img_feature_map.size()[3]

        img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2),2,3)
        f_wh_feature = img_feature_map.contiguous().view(batch_size*convsize*convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*convsize*convsize, 1, -1).repeat(1, self.num_classes, 1)

        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, 1024).repeat(batch_size*convsize*convsize,1,1)
        lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1,1024))
        coefficient = self.fc_a(lb_feature)
        coefficient = torch.transpose(torch.transpose(coefficient.view(batch_size, convsize, convsize, self.num_classes),2,3),1,2).view(batch_size, self.num_classes, -1)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, self.num_classes, convsize, convsize)
        coefficient = torch.transpose(torch.transpose(coefficient,1,2),2,3)
        coefficient = coefficient.view(batch_size, convsize, convsize, self.num_classes, 1).repeat(1,1,1,1,self.image_feature_dim)
        img_feature_map = img_feature_map.view(batch_size, convsize, convsize, 1, self.image_feature_dim).repeat(1, 1, 1, self.num_classes, 1)* coefficient
        graph_net_input = torch.sum(torch.sum(img_feature_map,1) ,1)
        """

        # new modified code for 1D/2D/3D input
        shape = img_feature_map.shape[2:]

        # project img_feature_map to f_wh_feature
        # [bz,c_img,shape] → [bz*shape, intermediary_dim] →repeat [bz*shape, label_num, intermediary_dim]
        f_wh_feature = img_feature_map.contiguous().view(-1, img_feature_map.shape[1])
        f_wh_feature = torch.unsqueeze(self.fc_1(f_wh_feature), 1).repeat(1, self.num_classes, 1)

        # project word_embedding to f_wd_feature
        # [label_num,c_word] → [label_num, intermediary_dim] →repeat [bz*shape, label_num, intermediary_dim]
        f_wd_feature = self.fc_2(word_embedding).view(1, self.num_classes, self.intermediary_dim).repeat(f_wh_feature.shape[0],1,1)

        # get the coefficient
        lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1, self.intermediary_dim))
        coefficient = self.fc_a(lb_feature)
        coefficient = coefficient.view(batch_size, self.num_classes, -1)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, *shape, self.num_classes)
        coefficient = torch.stack([coefficient for _ in range(self.image_feature_dim)], -1)
        img_feature_map = torch.stack([img_feature_map.view(batch_size, *shape, self.image_feature_dim) for _ in range(self.num_classes)], -2) * coefficient #
        graph_net_input = torch.sum(img_feature_map, dim=[ii+1 for ii in range(len(shape))])
        return graph_net_input


class SSGRL(nn.Module):
    def __init__(self, label_category_dict, image_channels, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_embedding, word_feature_dim=300, dim=2):
        """
        :param image_channels:
        :param image_feature_dim:
        :param output_dim:
        :param time_step:
        :param adjacency_matrix:
        :param word_embedding:
        :param num_classes:
        :param word_feature_dim:
        """
        super(SSGRL, self).__init__()

        num_labels = len(label_category_dict.keys())

        f_channel_list = [64, 128, 256, image_feature_dim]
        self.encoder = VGGEncoder(
            in_channels=image_channels,
            stage_output_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            downsample_ration=[0.5, 0.5, 0.5, 0.5],
            dim=dim)

        self.time_step = time_step
        self.num_labels = num_labels
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim

        self.word_semantic = semantic(num_classes=self.num_labels,
                                      image_feature_dim=self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.word_embedding = torch.tensor(word_embedding.astype(np.float32))
        self.adjacency_matrix = adjacency_matrix

        self.graph_net = GGNN(input_dim=self.image_feature_dim,
                              time_step=self.time_step,
                              in_matrix=torch.tensor(adjacency_matrix.astype(np.float32)),
                              out_matrix=torch.tensor(adjacency_matrix.astype(np.float32)))

        self.output_dim = output_dim
        self.fc_output = nn.Linear(2 * self.image_feature_dim, self.output_dim)
        self.classifiers = ClassificationHead(label_category_dict, self.output_dim, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        img_feature_map = self.encoder(x)[-1]
        graph_net_input = self.word_semantic(batch_size,
                                             img_feature_map,
                                             self.word_embedding)
        graph_net_feature = self.graph_net(graph_net_input)

        output = torch.cat((graph_net_feature.view(batch_size * self.num_labels, -1),
                            graph_net_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_labels, self.output_dim)
        output = torch.chunk(output,output.shape[1], 1)
        output = [i.view(batch_size, self.output_dim) for i in output]
        result = self.classifiers(output)
        return result


if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)

    # show dataset
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
            _l = list(label.split('_'))
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

    # print
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

    # which is slightly different from ML_GCN
    mode = 'without_self'  # or 'without_self'
    """
    without_self, which is the same as ML-GCN
    with_self, which is the only for VG dataset in SSGRL, 
    """
    for index in range(all_labels.shape[0]):
        _labels = all_labels[index]
        for i in range(all_labels.shape[1]):
            if _labels[i] == 1:
                nums_matrix[i] += 1
                for j in range(all_labels.shape[1]):
                    if mode == 'without_self': # ML-GCN implement
                        if j != i and _labels[j] == 1:
                            adj_matrix[i][j] += 1
                    elif mode == 'with_self':  # VG dataset in SSGRL implement
                        if _labels[j] == 1 and j == i:
                            adj_matrix[i][j] += 1
                        elif _labels[j] == 1 and j > i:
                            adj_matrix[i][j] += 1
                            adj_matrix[j][i] += 1

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

    # norm
    adjacency_matrix = (adj_matrix.T / nums_matrix).T  # every row refers to a list of conditional probability

    # 1D model test
    input = image_1D_tensor
    dim = 1
    print('-' * 22, 'build 1D model and test', '-'*18)
    print('input image batch shape:', input.shape)
    model = SSGRL(
        label_category_dict=label_category_dict,
        image_channels=input.shape[1],
        image_feature_dim=123,
        output_dim=2048,
        time_step=3,
        adjacency_matrix=adjacency_matrix,
        word_embedding=word_embedding,
        word_feature_dim=word_embedding.shape[1],
        dim=dim,
    )
    pre_logits_dict = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 2D model test
    input = image_2D_tensor
    dim = 2
    print('-' * 22, 'build 2D model and test', '-'*18)
    print('input image batch shape:', input.shape)
    model = SSGRL(
        label_category_dict=label_category_dict,
        image_channels=input.shape[1],
        image_feature_dim=123,
        output_dim=2048,
        time_step=3,
        adjacency_matrix=adjacency_matrix,
        word_embedding=word_embedding,
        word_feature_dim=word_embedding.shape[1],
        dim=dim,
    )
    pre_logits_dict = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 3D model test
    input = image_3D_tensor
    dim = 3
    print('-' * 22, 'build 3D model and test', '-'*18)
    print('input image batch shape:', input.shape)
    model = SSGRL(
        label_category_dict=label_category_dict,
        image_channels=input.shape[1],
        image_feature_dim=123,
        output_dim=2048,
        time_step=3,
        adjacency_matrix=adjacency_matrix,
        word_embedding=word_embedding,
        word_feature_dim=word_embedding.shape[1],
        dim=dim,
    )
    pre_logits_dict = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]
