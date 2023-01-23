# reference code: https://github.com/Alibaba-MIIL/ML_Decoder

import numpy as np
import torch
from torch import nn
from wama_modules.Encoder import VGGEncoder
from wama_modules.Transformer import TransformerDecoderLayer
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


class GroupFC():
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


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


class MLDecoder(nn.Module):
    def __init__(self, label_category_dict, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048):
        super().__init__()
        self.label_category_dict = label_category_dict
        num_classes = np.sum([label_category_dict[label] for label in label_category_dict.keys()])
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        self.embed_standart = embed_standart

        # non-learnable queries
        query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        query_embed.requires_grad_(False)
        self.query_embed = query_embed

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        self.decoder = TransformerDecoder(
            token_channels=decoder_embedding,
            depth=num_layers_decoder,
            heads=8,
            dim_head=decoder_embedding,
            mlp_dim=dim_feedforward,
            dropout=decoder_dropout)

        # group fully-connected
        self.decoder.num_classes = num_classes
        self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.decoder.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
        self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)

    def forward(self, x):
        bs = x.shape[0]  # batchsize
        if len(x.shape) >= 4:  # for 2D/3D
            embedding_spatial = x.view(bs, x.shape[1], -1).transpose(1, 2)
        else:
            embedding_spatial = x.transpose(1, 2)
        print(embedding_spatial.shape)
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        query_embed = self.query_embed.weight

        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)  # no allocation of memory with expand
        h, _, _ = self.decoder(tgt, embedding_spatial_786)  # [embed_len_decoder, batch, 768]

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)

        h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]

        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        pre_logits_dict = {}
        index = 0
        for label in label_category_dict.keys():
            pre_logits_dict[label] = logits[:, index:index + label_category_dict[label]]
            index += label_category_dict[label]

        return pre_logits_dict


if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)

    # 1D model test
    input = image_1D_tensor
    dim = 1
    print('-' * 22, 'build 1D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    encoder = VGGEncoder(
        in_channels=2,
        stage_output_channels=[64, 128, 256, 2048],
        blocks=[1, 2, 3, 4],
        downsample_ration=[0.5, 0.5, 0.5, 0.5],
        dim=dim)
    decoder = MLDecoder(
        label_category_dict=label_category_dict,
        num_of_groups=-1,
        decoder_embedding=768,
        initial_num_features=2048,
    )
    f = encoder(input)
    pre_logits_dict = decoder(f[-1])
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 2D model test
    input = image_2D_tensor
    dim = 2
    print('-' * 22, 'build 2D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    encoder = VGGEncoder(
        in_channels=2,
        stage_output_channels=[64, 128, 256, 2048],
        blocks=[1, 2, 3, 4],
        downsample_ration=[0.5, 0.5, 0.5, 0.5],
        dim=dim)
    decoder = MLDecoder(
        label_category_dict=label_category_dict,
        num_of_groups=-1,
        decoder_embedding=768,
        initial_num_features=2048,
    )
    f = encoder(input)
    pre_logits_dict = decoder(f[-1])
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]

    # 3D model test
    input = image_3D_tensor
    dim = 3
    print('-' * 22, 'build 3D model and test', '-'*18)
    print('input image batch shape:', input.shape)

    encoder = VGGEncoder(
        in_channels=2,
        stage_output_channels=[64, 128, 256, 2048],
        blocks=[1, 2, 3, 4],
        downsample_ration=[0.5, 0.5, 0.5, 0.5],
        dim=dim)
    decoder = MLDecoder(
        label_category_dict=label_category_dict,
        num_of_groups=-1,
        decoder_embedding=768,
        initial_num_features=2048,
    )
    f = encoder(input)
    pre_logits_dict = decoder(f[-1])
    _ = [print('logits of ', key, ':', pre_logits_dict[key].shape) for key in pre_logits_dict.keys()]