import torch.nn as nn
import torch
from torch import optim
from wama_modules.Transformer import TransformerEncoderLayer, TransformerDecoderLayer
from wama_modules.BaseModule import MakeNorm, GlobalMaxPool
from wama_modules.Decoder import UNet_decoder
from wama_modules.Head import SegmentationHead
from wama_modules.utils import load_weights, resizeTensor, tmp_class, tensor2array
from wama_modules.thirdparty_lib.MedicalNet_Tencent.model import generate_model
from wama_modules.PositionEmbedding import PositionalEncoding_1D_learnable
import matplotlib.pyplot as plt
import numpy as np

def show2D(img):
    plt.imshow(img)
    plt.show()


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


class RalationNet(nn.Module):
    def __init__(self,
                 organ_num=16, # actually, it should be organ_num+1(background)
                 encoder_weights_pth=None,  # encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth"
                 attention_type='cross',  # cross or self
                 relation_layer = 2,
                 relation_head = 8,
                 add_organ_embeddings = False,
                 dim=3):
        super().__init__()
        # self = tmp_class()  # for debug
        self.organ_num = organ_num
        self.attention_type = attention_type
        self.add_organ_embeddings = add_organ_embeddings
        self.dim = dim

        # encoder from thirdparty_lib.MedicalNet_Tencent
        Encoder_f_channel_list = [64, 64, 128, 256, 512]
        self.encoder = generate_model(18)
        if encoder_weights_pth is not None:
            pretrain_weights = torch.load(encoder_weights_pth, map_location='cpu')['state_dict']
            self.encoder = load_weights(self.encoder, pretrain_weights, drop_modelDOT=True, silence=True)

        # decoder for lesion
        Decoder_f_channel_list = [32, 64, 128, 256]
        self.decoder = UNet_decoder(
            in_channels_list=Encoder_f_channel_list,
            skip_connection=[True, True, True, True],
            out_channels_list=Decoder_f_channel_list,
            dim=dim)

        # seg head for lesion
        self.seg_head_lesion = SegmentationHead(
            label_category_dict=dict(lesion=1),
            in_channel=Decoder_f_channel_list[0],
            dim=dim)

        # seg head for organ
        self.seg_head_organ = SegmentationHead(
            label_category_dict=dict(organ=organ_num),
            in_channel=Encoder_f_channel_list[-1],  # encoder, not decoder
            dim=dim)

        # pool
        self.pool = GlobalMaxPool()

        # organ emb (optional)
        if add_organ_embeddings:
            self.organ_embed = PositionalEncoding_1D_learnable(
                embedding_dim=Encoder_f_channel_list[-1],
                token_num=organ_num)
        else:
            self.organ_embed = None

        # relation_module
        if attention_type == 'cross':
            self.relation_module = TransformerDecoder(
                token_channels=Encoder_f_channel_list[-1],
                depth=relation_layer,
                heads=relation_head,
                dim_head=Encoder_f_channel_list[-1],
                mlp_dim=Encoder_f_channel_list[-1],
            )
        elif attention_type == 'self':
            self.norm = MakeNorm(dim, Encoder_f_channel_list[-1], norm='ln')
            self.relation_module = TransformerEncoder(
                token_channels=Encoder_f_channel_list[-1],
                depth=relation_layer,
                heads=relation_head,
                dim_head=Encoder_f_channel_list[-1],
                mlp_dim=Encoder_f_channel_list[-1],
            )
        else:
            raise ValueError('must be cross or self')

    def forward(self, x):
        bz = x.shape[0]

        # achieve encoder feaetures
        f_encoder = self.encoder(x)  # _ = [print(i.shape) for i in f_encoder]

        # achieve organ seg map from the deepest encoder feature (called deep supervision)
        organ_seg_map = self.seg_head_organ(f_encoder[-1])  # _ = [print(key, organ_seg_map[key].shape) for key in organ_seg_map.keys()]
        organ_seg_map_tensor = organ_seg_map['organ']
        organ_seg_map_tensor = organ_seg_map_tensor.contiguous().view(bz, organ_seg_map_tensor.shape[1],-1)
        organ_seg_map_tensor = torch.softmax(organ_seg_map_tensor, dim=1)  # print(organ_seg_map_tensor.sum(1).shape, organ_seg_map_tensor.sum(1))
        organ_seg_map_tensor = organ_seg_map_tensor.contiguous().view(*organ_seg_map['organ'].shape)

        # achieve organ tokens by multiplying the segmentation map by the feature graph
        organ_tokens = [self.pool(torch.unsqueeze(organ_seg_map_tensor[:,i],1)*f_encoder[-1]) for i in range(self.organ_num)]  # _ = [print(i.shape) for i in organ_tokens]

        # achieve lesion token
        lesion_token = self.pool(f_encoder[-1])
        print(lesion_token.shape)

        # relation module forward
        if self.attention_type == 'cross':
            lesion_token, self_attn_map_list, cross_attn_map_list = self.relation_module(
                torch.unsqueeze(lesion_token,1),
                torch.stack(organ_tokens, dim=1),
                q_pos_embeddings=None,
                v_pos_embeddings=self.organ_embed)
            lesion_token = lesion_token.view(bz, lesion_token.shape[2],1,1,1)
            lesion_token = lesion_token.repeat(1, 1, *f_encoder[-1].shape[2:])
        elif self.attention_type == 'self':
            organ_tokens = torch.stack(organ_tokens, dim=1)
            if self.add_organ_embeddings:
                print('add organ emb')
                organ_tokens += self.organ_embed
            all_tokens = torch.cat([organ_tokens, torch.unsqueeze(lesion_token,1)], dim=1)
            all_tokens, self_attn_map_list = self.relation_module(self.norm(all_tokens), pos_emb=None)
            lesion_token = all_tokens[:,-1,:]
            lesion_token = lesion_token.view(bz, lesion_token.shape[1],1,1,1)
            lesion_token = lesion_token.repeat(1, 1, *f_encoder[-1].shape[2:])

        # get lesion seg map
        f_decoder = self.decoder(f_encoder[:-1]+[lesion_token])
        f_for_seg = resizeTensor(f_decoder[0], size=x.shape[2:])
        lesion_seg_map = self.seg_head_lesion(f_for_seg)


        return lesion_seg_map, organ_seg_map, organ_seg_map_tensor


class RalationNet_v2(nn.Module):
    def __init__(self,
                 organ_num=16, # actually, it should be organ_num+1(background)
                 encoder_weights_pth=None,  # encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth"
                 attention_type='cross',  # cross or self
                 relation_layer = 2,
                 relation_head = 8,
                 add_organ_embeddings = False,
                 dim=3):
        super().__init__()
        # self = tmp_class()  # for debug
        self.organ_num = organ_num
        self.attention_type = attention_type
        self.add_organ_embeddings = add_organ_embeddings
        self.dim = dim

        # encoder from thirdparty_lib.MedicalNet_Tencent
        Encoder_f_channel_list = [64, 64, 128, 256, 512]
        self.encoder = generate_model(18)
        if encoder_weights_pth is not None:
            pretrain_weights = torch.load(encoder_weights_pth, map_location='cpu')['state_dict']
            self.encoder = load_weights(self.encoder, pretrain_weights, drop_modelDOT=True, silence=True)

        # decoder for lesion
        Decoder_f_channel_list = [32, 64, 128, 256]
        self.decoder = UNet_decoder(
            in_channels_list=Encoder_f_channel_list,
            skip_connection=[True, True, True, True],
            out_channels_list=Decoder_f_channel_list,
            dim=dim)

        # seg head for lesion
        self.seg_head_lesion = SegmentationHead(
            label_category_dict=dict(lesion=1),
            in_channel=Decoder_f_channel_list[0],
            dim=dim)

        # seg head for organ
        self.seg_head_organ = SegmentationHead(
            label_category_dict=dict(organ=organ_num),
            in_channel=Encoder_f_channel_list[-1],  # encoder, not decoder
            dim=dim)

        # pool
        self.pool = GlobalMaxPool()

        # organ emb (optional)
        if add_organ_embeddings:
            self.organ_embed = PositionalEncoding_1D_learnable(
                embedding_dim=Encoder_f_channel_list[-1],
                token_num=organ_num)
        else:
            self.organ_embed = None

        # relation_module
        if attention_type == 'cross':
            self.relation_module = TransformerDecoder(
                token_channels=Encoder_f_channel_list[-1],
                depth=relation_layer,
                heads=relation_head,
                dim_head=Encoder_f_channel_list[-1],
                mlp_dim=Encoder_f_channel_list[-1],
            )
        elif attention_type == 'self':
            self.norm = MakeNorm(dim, Encoder_f_channel_list[-1], norm='ln')
            self.relation_module = TransformerEncoder(
                token_channels=Encoder_f_channel_list[-1],
                depth=relation_layer,
                heads=relation_head,
                dim_head=Encoder_f_channel_list[-1],
                mlp_dim=Encoder_f_channel_list[-1],
            )
        else:
            raise ValueError('must be cross or self')

    def forward(self, x):
        bz = x.shape[0]

        # achieve encoder feaetures
        f_encoder = self.encoder(x)  # _ = [print(i.shape) for i in f_encoder]

        # achieve organ seg map from the deepest encoder feature (called deep supervision)
        organ_seg_map = self.seg_head_organ(f_encoder[-1])  # _ = [print(key, organ_seg_map[key].shape) for key in organ_seg_map.keys()]
        organ_seg_map_tensor = organ_seg_map['organ']
        organ_seg_map_tensor = organ_seg_map_tensor.contiguous().view(bz, organ_seg_map_tensor.shape[1],-1)
        organ_seg_map_tensor = torch.softmax(organ_seg_map_tensor, dim=1)  # print(organ_seg_map_tensor.sum(1).shape, organ_seg_map_tensor.sum(1))
        organ_seg_map_tensor = organ_seg_map_tensor.contiguous().view(*organ_seg_map['organ'].shape)

        # achieve organ tokens by multiplying the segmentation map by the feature graph
        organ_tokens = [self.pool(torch.unsqueeze(organ_seg_map_tensor[:,i],1)*f_encoder[-1]) for i in range(self.organ_num)]  # _ = [print(i.shape) for i in organ_tokens]

        # achieve lesion token
        lesion_token = f_encoder[-1].contiguous().view(bz, -1, f_encoder[-1].shape[1])
        print(lesion_token.shape)

        # relation module forward
        if self.attention_type == 'cross':
            lesion_token, self_attn_map_list, cross_attn_map_list = self.relation_module(
                lesion_token,
                torch.stack(organ_tokens, dim=1),
                q_pos_embeddings=None,
                v_pos_embeddings=self.organ_embed)
            lesion_token = lesion_token.contiguous().view(*f_encoder[-1].shape)
        elif self.attention_type == 'self':
            organ_tokens = torch.stack(organ_tokens, dim=1)
            if self.add_organ_embeddings:
                print('add organ emb')
                organ_tokens += self.organ_embed
            all_tokens = torch.cat([lesion_token,organ_tokens], dim=1)
            all_tokens, self_attn_map_list = self.relation_module(self.norm(all_tokens), pos_emb=None)
            lesion_token = all_tokens[:,:lesion_token.shape[1],:]
            lesion_token = lesion_token.contiguous().view(*f_encoder[-1].shape)

        # get lesion seg map
        f_decoder = self.decoder(f_encoder[:-1]+[lesion_token])
        f_for_seg = resizeTensor(f_decoder[0], size=x.shape[2:])
        lesion_seg_map = self.seg_head_lesion(f_for_seg)

        return lesion_seg_map, organ_seg_map, organ_seg_map_tensor


if __name__ == '__main__':
    batchsize = 2
    channel = 1
    shape = [256,256,16]
    organ_num = 16+1  # 1 refer to background
    image = torch.ones([batchsize,channel]+shape)
    image[:,:,:64,:64,:16] = 0
    organ_GT = resizeTensor(torch.ones([batchsize,organ_num]+shape), size=[32,32,2])
    lesion_GT = torch.ones([batchsize,1]+shape)

    RalationNet = RalationNet_v2  # use this to switch to RalationNet_v2

    # mode1: self attention w/ organ emb ------------------------
    print('# mode1: self attention w/ organ emb ------------------------')
    model = RalationNet(
        organ_num=organ_num,  # actually, it should be organ_num+1(background)
        encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth",
        attention_type='self',  # cross or self
        relation_layer=2,
        relation_head=8,
        add_organ_embeddings=True,
    )
    optimized_parameters = list(model.parameters())
    optimizer = optim.Adam(optimized_parameters, 1e-3, [0.5, 0.999], weight_decay= 5e-4)

    lesion_seg_map, organ_seg_map, organ_seg_map_tensor= model(image)
    loss = ((lesion_seg_map['lesion']-lesion_GT)**2).sum() + ((organ_seg_map['organ']-organ_GT)**2).sum()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # mode2: self attention w/o organ emb ------------------------
    print('# mode2: self attention w/o organ emb ------------------------')
    model = RalationNet(
        organ_num=organ_num,  # actually, it should be organ_num+1(background)
        encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth",
        attention_type='self',  # cross or self
        relation_layer=2,
        relation_head=8,
        add_organ_embeddings=False,
    )
    optimized_parameters = list(model.parameters())
    optimizer = optim.Adam(optimized_parameters, 1e-3, [0.5, 0.999], weight_decay= 5e-4)

    lesion_seg_map, organ_seg_map, organ_seg_map_tensor = model(image)
    loss = ((lesion_seg_map['lesion']-lesion_GT)**2).sum() + ((organ_seg_map['organ']-organ_GT)**2).sum()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # mode3: cross attention w/ organ emb ------------------------
    print('# mode3: cross attention w/ organ emb ------------------------')
    model = RalationNet(
        organ_num=organ_num,  # actually, it should be organ_num+1(background)
        encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth",
        attention_type='cross',  # cross or self
        relation_layer=2,
        relation_head=8,
        add_organ_embeddings=True,
    )
    optimized_parameters = list(model.parameters())
    optimizer = optim.Adam(optimized_parameters, 1e-3, [0.5, 0.999], weight_decay= 5e-4)

    lesion_seg_map, organ_seg_map, organ_seg_map_tensor = model(image)
    loss = ((lesion_seg_map['lesion']-lesion_GT)**2).sum() + ((organ_seg_map['organ']-organ_GT)**2).sum()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # mode4: cross attention w/o organ emb ------------------------
    print('# mode4: cross attention w/o organ emb ------------------------')
    model = RalationNet(
        organ_num=organ_num,  # actually, it should be organ_num+1(background)
        encoder_weights_pth=r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth",
        attention_type='cross',  # cross or self
        relation_layer=2,
        relation_head=8,
        add_organ_embeddings=False,
    )
    optimized_parameters = list(model.parameters())
    optimizer = optim.Adam(optimized_parameters, 1e-3, [0.5, 0.999], weight_decay= 5e-4)

    lesion_seg_map, organ_seg_map, organ_seg_map_tensor = model(image)
    loss = ((lesion_seg_map['lesion']-lesion_GT)**2).sum() + ((organ_seg_map['organ']-organ_GT)**2).sum()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # vis to check whether the tensor.view is correct for keeping spacing information
    lesion_seg_map = tensor2array(lesion_seg_map['lesion'])
    show2D(np.squeeze(lesion_seg_map[0,0,0]))
    show2D(np.squeeze(lesion_seg_map[0,0,:,0]))
    show2D(np.squeeze(lesion_seg_map[0,0,:,:,0]))

    organ_seg_map = tensor2array(organ_seg_map['organ'])
    show2D(np.squeeze(organ_seg_map[0,0,0]))
    show2D(np.squeeze(organ_seg_map[0,0,:,0]))
    show2D(np.squeeze(organ_seg_map[0,0,:,:,0]))
    show2D(np.squeeze(organ_seg_map[0,1,:,:,0]))
    show2D(np.squeeze(organ_seg_map[0,2,:,:,0]))

    organ_seg_map = tensor2array(organ_seg_map_tensor)  # after sigmoid
    show2D(np.squeeze(organ_seg_map[0,1,0]))
    show2D(np.squeeze(organ_seg_map[0,1,:,0]))
    show2D(np.squeeze(organ_seg_map[0,0,:,:,0]))
    show2D(np.squeeze(organ_seg_map[0,1,:,:,0]))
    show2D(np.squeeze(organ_seg_map[0,2,:,:,0]))