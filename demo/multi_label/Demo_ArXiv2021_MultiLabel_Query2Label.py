# reference code:https://github.com/SlongLiu/query2labels
# BTW, the Q2L code is modified from DETR
# details:
#   1) in both encoder and decoder, there is NO position embeddings added to Value, but added to key
#   2) ----- in original implement,
#      the label embedding is initialized with zero,
#      and its position embedding is the true label embedding
#      https://github.com/SlongLiu/query2labels/blob/55eb05064f4badbe03423b79e5c9d143da2dff2e/lib/models/transformer.py#L112
#      ----- which is the same as:
#      the label embedding is initialized with true label embedding,and its position embedding is None
#      this script will implement in the @2nd way
#   3) the structure is simple: first the image features f will go through n-layer encoder to generate f~,
#       and then f~ and label embeddings will together go through the decoder


import numpy as np
import torch.nn as nn
import torch
from wama_modules.Transformer import TransformerEncoderLayer, TransformerDecoderLayer
from wama_modules.PositionEmbedding import PositionalEncoding_2D_sincos,PositionalEncoding_3D_sincos,PositionalEncoding_1D_sincos
from wama_modules.Encoder import ResNetEncoder
from wama_modules.BaseModule import MakeNorm
from wama_modules.Head import ClassificationHead
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


class TransformerEncoder(nn.Module):
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
                AddPosEmb2Value=False,
            ) for _ in range(depth)])

    def forward(self, tokens, pos_emb):
        """

        :param tokens: tensor with shape of [batchsize, token_num, token_channels]
        :return: tokens, attn_map_list

        # demo
        token_channels = 512
        token_num = 5
        batchsize = 3
        depth = 3
        heads = 8
        dim_head = 32
        mlp_dim = 64

        tokens = torch.ones([batchsize, token_num, token_channels])
        pos_emb = torch.ones([batchsize, token_num, token_channels])

        encoder = TransformerEncoder(token_channels, depth, heads, dim_head, mlp_dim=mlp_dim, dropout=0.)
        tokens_, attn_map_list = encoder(tokens, pos_emb)
        print(tokens.shape, tokens_.shape)
        _ = [print(i.shape) for i in attn_map_list]
        """
        attn_map_list = []
        for layer in self.layers:
            tokens, attention_maps = layer(tokens, pos_emb)
            attn_map_list.append(attention_maps)  # from shallow to deep
        return tokens, attn_map_list


class TransformerDecoder(nn.Module):
    def __init__(self, token_channels, depth, heads, dim_head, mlp_dim=None, dropout=0.):
        """
        :param depth: number of layers
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                token_channels,
                heads=heads,
                dim_head=dim_head,
                channel_mlp=mlp_dim,
                dropout=dropout,
                AddPosEmb2Value=False,
            ) for _ in range(depth)])

    def forward(self, q_tokens, v_tokens, q_pos_embeddings=None, v_pos_embeddings=None):
        """

        :param tokens: tensor with shape of [batchsize, token_num, token_channels]
        :return: q_tokens, attn_map_list

        # demo
        token_channels = 512
        q_token_num = 5
        v_token_num = 10
        batchsize = 3
        depth = 3
        heads = 8
        dim_head = 32
        mlp_dim = 64

        q_tokens = torch.ones([batchsize, q_token_num, token_channels])

        v_tokens = torch.ones([batchsize, v_token_num, token_channels])
        v_pos_emb = torch.ones([batchsize, v_token_num, token_channels])

        decoder = TransformerDecoder(token_channels, depth, heads, dim_head, mlp_dim=mlp_dim, dropout=0.)
        q_tokens_, self_attn_map_list, cross_attn_map_list = decoder(q_tokens, v_tokens, None, v_pos_emb)
        print(q_tokens.shape, q_tokens_.shape)
        _ = [print(i.shape) for i in self_attn_map_list]
        _ = [print(i.shape) for i in cross_attn_map_list]
        """
        self_attn_map_list = []
        cross_attn_map_list = []
        for layer in self.layers:
            q_tokens, self_attn_map, cross_attn_map = layer(q_tokens, v_tokens, q_pos_embeddings, v_pos_embeddings)
            self_attn_map_list.append(self_attn_map)  # from shallow to deep
            cross_attn_map_list.append(cross_attn_map)  # from shallow to deep
        return q_tokens, self_attn_map_list, cross_attn_map_list


class Q2L(nn.Module):
    def __init__(self,
                 label_category_dict,
                 in_channel=1,
                 position_embedding=True,  # default is False, see https://github.com/QData/C-Tran/issues/12
                 encoder_transformer_depth=1,  # default is 1
                 decoder_transformer_depth=2,  # default is 2
                 transformer_heads=4,  # default is 4
                 dim=2):
        super().__init__()
        # self = tmp_class()
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
        self.num_labels = len(label_name)
        self.label_input = torch.Tensor(np.arange(self.num_labels)).view(1, -1).long()
        self.label_embed = torch.nn.Embedding(self.num_labels, f_channel_list[-1], padding_idx=None)
        # self.label_embed(torch.tensor([2,2,2,2,2]))

        # Normalization for tokens
        self.norm = MakeNorm(dim, f_channel_list[-1], norm='ln')

        # Transformer Encoder
        self.transEncoder = TransformerEncoder(
            token_channels=f_channel_list[-1],
            depth=encoder_transformer_depth,
            heads=transformer_heads,
            dim_head=f_channel_list[-1],
            mlp_dim=f_channel_list[-1],
        )

        # Transformer Decoder
        self.transDecoder = TransformerDecoder(
            token_channels=f_channel_list[-1],
            depth=decoder_transformer_depth,
            heads=transformer_heads,
            dim_head=f_channel_list[-1],
            mlp_dim=f_channel_list[-1],
        )

        # cls head
        self.cls_head = ClassificationHead(label_category_dict, f_channel_list[-1], bias=True)

    def forward(self, image):
        """
        in inference phase, the label_value_dict and label_known_dict could be set None
        :param image: [bz, channel, *shape]
        :param label_value_dict: see demo format
        """

        batchsize = image.shape[0]

        # extract image embeddings
        image_tokens = self.img_embed(image)[-1]
        if self.position_embedding:
            print('add position embeddings')
            if self.dim == 1:
                pos_emb = PositionalEncoding_1D_sincos(embedding_dim=image_tokens.shape[1], token_num=image_tokens.shape[2])
            elif self.dim == 2:
                pos_emb = PositionalEncoding_2D_sincos(embedding_dim=image_tokens.shape[1], token_shape=image_tokens.shape[2:])
            elif self.dim == 3:
                pos_emb = PositionalEncoding_3D_sincos(embedding_dim=image_tokens.shape[1], token_shape=image_tokens.shape[2:])

            pos_emb = pos_emb.view(1, image_tokens.size(1), -1).permute(0, 2, 1)
        else:
            pos_emb = None
        image_tokens = image_tokens.view(image_tokens.size(0), image_tokens.size(1), -1).permute(0, 2, 1)  # [bz, token_num, channel]

        # extract label embeddings
        label_tokens = self.label_embed(self.label_input).repeat(batchsize, 1, 1)

        # Encoder Transformer forward
        image_tokens, _ = self.transEncoder(self.norm(image_tokens), pos_emb)

        # Decoder Transformer forward
        label_tokens, self_attn_map_list, cross_attn_map_list = self.transDecoder(
            label_tokens, image_tokens, q_pos_embeddings=None, v_pos_embeddings=pos_emb)

        # Cls head forward
        output_label_tokens = label_tokens
        output_label_tokens = torch.chunk(output_label_tokens, output_label_tokens.shape[1], 1)
        output_label_tokens = [i.view(i.shape[0], i.shape[-1]) for i in output_label_tokens]
        predict_logits_dict = self.cls_head(output_label_tokens)

        return predict_logits_dict, self_attn_map_list, cross_attn_map_list


if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)

    # 1D Q2L model
    input = image_1D_tensor
    print('-' * 22, 'build 1D model', '-'*18)
    print('input image batch shape:', input.shape)
    model = Q2L(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,  # default is True, see https://github.com/SlongLiu/query2labels/blob/main/main_mlc.py#L142
        encoder_transformer_depth=1,  # default is 1
        decoder_transformer_depth=2,  # default is 2
        transformer_heads=4,  # default is 4
        dim=1
    )
    pre_logits_dict, self_attn_map_list, cross_attn_map_list = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 2D Q2L model
    input = image_2D_tensor
    print('-' * 22, 'build 2D model', '-'*18)
    print('input image batch shape:', input.shape)
    model = Q2L(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,  # default is True, see https://github.com/SlongLiu/query2labels/blob/main/main_mlc.py#L142
        encoder_transformer_depth=1,  # default is 1
        decoder_transformer_depth=2,  # default is 2
        transformer_heads=4,  # default is 4
        dim=2
    )
    pre_logits_dict, self_attn_map_list, cross_attn_map_list = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]


    # 3D Q2L model
    input = image_3D_tensor
    print('-' * 22, 'build 3D model', '-'*18)
    print('input image batch shape:', input.shape)
    model = Q2L(
        label_category_dict,
        in_channel=input.shape[1],
        position_embedding=True,  # default is True, see https://github.com/SlongLiu/query2labels/blob/main/main_mlc.py#L142
        encoder_transformer_depth=1,  # default is 1
        decoder_transformer_depth=2,  # default is 2
        transformer_heads=4,  # default is 4
        dim=3
    )
    pre_logits_dict, self_attn_map_list, cross_attn_map_list = model(input)
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]
