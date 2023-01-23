# reference code: https://github.com/QData/C-Tran

import numpy as np
import torch.nn as nn
import torch
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Transformer import TransformerEncoderLayer
from wama_modules.PositionEmbedding import PositionalEncoding_2D_sincos, PositionalEncoding_3D_sincos, PositionalEncoding_1D_sincos
from wama_modules.BaseModule import MakeNorm
from wama_modules.Head import ClassificationHead
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


# build the model C-tran
class TransformerEncoder(nn.Module):
    # for C-tran
    def __init__(self, token_channels, depth, heads, dim_head, mlp_dim=None, dropout=0.):
        """
        :param depth: number of layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                token_channels,
                heads=heads,
                dim_head=dim_head,
                channel_mlp=mlp_dim,
                dropout=dropout,
            ) for _ in range(depth)])

    def forward(self, tokens):
        """

        :param tokens: tensor with shape of [batchsize, token_num, token_channels]
        :return: tokens, attn_map_list

        # demo1: no position_embedding --------------------------------------------
        token_channels = 512
        token_num = 5
        batchsize = 3
        depth = 3
        heads = 8
        dim_head = 32
        mlp_dim = 64

        tokens = torch.ones([batchsize, token_num, token_channels])

        encoder = TransformerEncoder(token_channels, depth, heads, dim_head, mlp_dim=mlp_dim, dropout=0.)
        tokens_, attn_map_list = encoder(tokens)
        print(tokens.shape, tokens_.shape)
        _ = [print(i.shape) for i in attn_map_list]
        """
        attn_map_list = []
        for layer in self.layers:
            tokens, attention_maps = layer(tokens)
            attn_map_list.append(attention_maps)  # from shallow to deep
        return tokens, attn_map_list


class C_Tran(nn.Module):
    def __init__(self,
                 label_category_dict,
                 in_channel=1,
                 position_embedding=False,  # default is False, see https://github.com/QData/C-Tran/issues/12
                 transformer_depth=3,  # default is 3
                 transformer_heads=4,  # default is 4
                 dim=2,  # 1D/2D/3D input
                 ):
        super().__init__()
        self.dim = dim
        self.label_category_dict = label_category_dict
        self.label_name = list(label_category_dict.keys())

        # Image Embeddings
        f_channel_list = [64, 128, 256, 6*128]
        self.img_embed = ResNetEncoder(
            in_channel,
            stage_output_channels=f_channel_list,
            stage_middle_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            type='131',
            downsample_ration=[0.5, 0.5, 0.5, 0.8],
            dim=dim)
        self.position_embedding = position_embedding

        # Label Embeddings
        self.num_labels = len(self.label_name)
        self.label_input = torch.Tensor(np.arange(self.num_labels)).view(1, -1).long()
        self.label_embed = torch.nn.Embedding(self.num_labels, f_channel_list[-1], padding_idx=None)
        # self.label_embed(torch.tensor([2,2,2,2,2]))

        # State Embeddings, [negative positive unknown]
        self.num_state = 3
        self.state_embed = torch.nn.Embedding(3, f_channel_list[-1], padding_idx=2)
        # self.state_embed(torch.tensor([2,2,2,2,2]))

        # Normalization for tokens
        self.norm = MakeNorm(dim, f_channel_list[-1], norm='ln')

        # Transformer
        self.transformer = TransformerEncoder(
            token_channels=f_channel_list[-1],
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=f_channel_list[-1],
            mlp_dim=f_channel_list[-1],
        )

        # cls head
        self.cls_head = ClassificationHead(label_category_dict, f_channel_list[-1], bias=True)

    def forward(self, image, label_value_dict=None, label_known_dict=None):
        """
        in inference phase, the label_value_dict and label_known_dict could be set None
        :param image: [bz, channel, *shape]
        :param label_value_dict: see demo format
        :param label_known_dict: see demo format (0=unknown/missing, 1=known)
        :return:
        """

        batchsize = image.shape[0]

        # inference phase
        if label_value_dict is None or label_known_dict is None:
            label_value_dict = {}
            label_known_dict = {}
            for label in self.label_name:
                label_value_dict[label] = torch.zeros([batchsize],dtype=torch.long).to(image.device)
                label_known_dict[label] = torch.zeros([batchsize],dtype=torch.long).to(image.device)

        # extract image embeddings
        f_image = self.img_embed(image)[-1]
        if self.position_embedding:
            print('add position embeddings')
            if self.dim == 1:
                pos_emb = PositionalEncoding_1D_sincos(embedding_dim=f_image.shape[1], token_num=f_image.shape[2])
            elif self.dim == 2:
                pos_emb = PositionalEncoding_2D_sincos(embedding_dim=f_image.shape[1], token_shape=f_image.shape[2:])
            elif self.dim == 3:
                pos_emb = PositionalEncoding_3D_sincos(embedding_dim=f_image.shape[1], token_shape=f_image.shape[2:])

            f_image += pos_emb.to(image.device)

        image_tokens = f_image.view(f_image.size(0), f_image.size(1), -1).permute(0, 2, 1)  # [bz, token_num, channel]

        # extract label embeddings
        label_tokens = self.label_embed(self.label_input).repeat(batchsize, 1, 1)

        # extract state embedding
        label_value = [label_value_dict[label] for label in self.label_name]
        known_mask = [label_known_dict[label] for label in self.label_name]
        state_tokens_list = []
        for label_index, label in enumerate(self.label_name):
            _labelValue = label_value[label_index]
            _knownMask = known_mask[label_index]  # 1 known 0 unknown (2 represents "unknown")
            for i in range(len(_labelValue)):
                if _knownMask[i] == 0:
                    _labelValue[i] = 2  # 2 represents "unknown"
            _state_embed = self.state_embed(_labelValue)
            state_tokens_list.append(_state_embed)
        state_tokens = torch.stack(state_tokens_list, 1)

        # Transformer forward
        input_tokens = torch.cat([state_tokens+label_tokens, image_tokens], 1)
        output_tokens, attn_map_list = self.transformer(self.norm(input_tokens))

        # Cls head forward
        output_label_tokens = output_tokens[:,:len(self.label_name)]
        output_label_tokens = torch.chunk(output_label_tokens, output_label_tokens.shape[1], 1)
        output_label_tokens = [i.view(i.shape[0], i.shape[-1]) for i in output_label_tokens]
        predict_logits_dict = self.cls_head(output_label_tokens)
        return predict_logits_dict, attn_map_list


if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)
    label_value_dict = {}
    label_known_dict = {}
    for label_index, label in enumerate(label_name):
        label_value_dict[label] = torch.tensor([case['label_value'][label_index] for case in dataset])
        label_known_dict[label] = torch.tensor([case['label_known'][label_index] for case in dataset])

    # build 1D model and test (w/o pos emb)
    input = image_1D_tensor
    print('-' * 22, 'build 1D model and test (w/o pos emb)', '-' * 22)
    print('input image batch shape:', input.shape)
    model = C_Tran(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=False,
        transformer_depth=3,
        transformer_heads=4,
        dim=1
    )
    pre_logits_dict, attention_list = model(input, label_value_dict, label_known_dict)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # build 2D model and test (w/o pos emb, which is the default setting)
    input = image_2D_tensor
    print('-' * 22, 'build 2D model and test (w/o pos emb)', '-' * 22)
    print('input image batch shape:', input.shape)
    model = C_Tran(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=False,
        transformer_depth=3,
        transformer_heads=4,
        dim=2
    )
    pre_logits_dict, attention_list = model(input, label_value_dict, label_known_dict)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # build 2D model and test model (w/ pos emb)
    input = image_2D_tensor
    print('-' * 22, 'build 2D model and test model (w/ pos emb)', '-'*18)
    print('input image batch shape:', input.shape)
    model = C_Tran(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,
        transformer_depth=3,
        transformer_heads=4,
        dim=2
    )
    pre_logits_dict, attention_list = model(input, label_value_dict, label_known_dict)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # build 3D model and test (w/o pos emb)
    input = image_3D_tensor
    print('-' * 22, 'build 3D model and test model (w/ pos emb)', '-'*18)
    print('input image batch shape:', input.shape)

    model = C_Tran(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,
        transformer_depth=3,
        transformer_heads=4,
        dim=3
    )
    pre_logits_dict, attention_list = model(input, label_value_dict, label_known_dict)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # build 3D model and test (w/o pos emb and label_value_dict)
    input = image_3D_tensor
    print('-' * 22, 'build 3D model and test model (w/ pos emb and label_value_dict)', '-'*18)
    print('input image batch shape:', input.shape)

    model = C_Tran(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,
        transformer_depth=3,
        transformer_heads=4,
        dim=3
    )
    pre_logits_dict, attention_list = model(input)  # in reference phase, w/o label_value_dict and label_known_dict
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]
