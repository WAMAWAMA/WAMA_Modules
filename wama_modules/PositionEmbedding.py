# 1D, 2D, 3D
import math
import torch
from torch import nn
from wama_modules.utils import tensor2array, mat2gray


def PositionalEncoding_1D_learnable(embedding_dim=1024, token_num=600):
    """
    return:  [1, token_num, token_dim]
    """
    return nn.Parameter(torch.zeros(1, token_num, embedding_dim))


def PositionalEncoding_1D_sincos(embedding_dim=1024, token_num=600, temperature=10000.):
    """
    from the original transformer "attention is all you need"
    :param embedding_dim:
    :param token_num:
    :param temperature:
    :return:  [1, token_num, token_dim]

    from wama_modules.utils import show2D
    pe = tensor2array(torch.squeeze(PositionalEncoding_1D_sincos(1002, 200, temperature=10000)))
    show2D(pe)
    pe = tensor2array(torch.squeeze(PositionalEncoding_1D_sincos(1002, 200, temperature=1000)))
    show2D(pe)

    """
    pe = torch.zeros(token_num, embedding_dim)
    position = torch.arange(0, token_num).unsqueeze(1)
    # e**(a*b) = (e**a)**b
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(temperature) / embedding_dim))  # term 10000**（2i/d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even number pe
    pe[:, 1::2] = torch.cos(position * div_term)  # odd number pe
    pe = pe.unsqueeze(0)
    return pe  # [1, token_num, token_dim]


def PositionalEncoding_2D_learnable(embedding_dim=1024, token_shape=[100, 200], return_flatten=False):
    """
    return:
        if return_flatten is True, [1, token_shape[0]*token_shape[1], token_dim]
        else,  [1, token_dim, *token_shape]

    # demo
    print(PositionalEncoding_2D_learnable(1024,[100,300]).shape)
    print(PositionalEncoding_2D_learnable(1024,[100,300],return_flatten = True).shape)

    """
    if return_flatten:
        return nn.Parameter(torch.zeros(1, embedding_dim, token_shape[0]*token_shape[1]))
    else:
        return nn.Parameter(torch.zeros(1, token_shape[0], token_shape[1], embedding_dim))


def PositionalEncoding_2D_sincos(embedding_dim=1024, token_shape = [100,200], temperature = 10000., return_flatten=False):
    """
    from C-tran:
    https://github.com/QData/C-Tran/blob/9278a693872a00dd7bdf8f6cdf2a599b83f01d16/models/position_enc.py#L59

    which is the same as MAE  or  moco and stuff:
    moco: https://github.com/facebookresearch/moco-v3/blob/38aae447f5e8f5a6221dbff0d8db40012512fe32/vits.py#L53
    mae: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20

    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return:
        if return_flatten is True, [1, token_shape[0]*token_shape[1], token_dim]
        else,  [1, token_dim, *token_shape]

    from wama_modules.utils import show3D
    print(PositionalEncoding_2D_sincos(1024, [200, 100]).shape)
    pe = tensor2array(torch.squeeze(PositionalEncoding_2D_sincos(1024, [200, 200])))
    show3D(pe)
    pe = tensor2array(torch.squeeze(PositionalEncoding_2D_sincos(1024, [200, 200], temperature=1000)))
    show3D(pe)

    pe = tensor2array(torch.squeeze(
    PositionalEncoding_2D_sincos(1024, [200, 200], temperature=1000, return_flatten=True).reshape(1,1024,200,200)
    ))
    show3D(pe)

    """
    if embedding_dim % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(embedding_dim))
    height, width = token_shape
    pe = torch.zeros(embedding_dim, height, width)
    # Each dimension use half of d_model
    or_embedding_dim = embedding_dim
    embedding_dim = int(embedding_dim / 2)
    div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(temperature) / embedding_dim))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:embedding_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:embedding_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[embedding_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[embedding_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.unsqueeze(0)

    if return_flatten:
        return pe.reshape(1, token_shape[0]*token_shape[1], or_embedding_dim)
    else:
        return pe


def PositionalEncoding_3D_learnable(embedding_dim=1024, token_shape=[20, 30, 40], return_flatten=False):
    """
    return:
        if return_flatten is True, [1, token_shape[0]*token_shape[1]*token_shape[2], token_dim]
        else,  [1, token_dim, *token_shape]

    # demo
    print(PositionalEncoding_3D_learnable(1024,[20, 30, 40]).shape)
    print(PositionalEncoding_3D_learnable(1024,[20, 30, 40],return_flatten = True).shape)

    """
    if return_flatten:
        return nn.Parameter(torch.zeros(1, embedding_dim, token_shape[0]*token_shape[1]*token_shape[2]))
    else:
        return nn.Parameter(torch.zeros(1, token_shape[0], token_shape[1], token_shape[2], embedding_dim))


def PositionalEncoding_3D_sincos(embedding_dim=6*128, token_shape = [20,30,40], temperature = 10000., return_flatten=False):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return:
        if return_flatten is True, [1, token_shape[0]*token_shape[1]*token_shape[2], token_dim]
        else,  [1, token_dim, *token_shape]

    # demo:

    from wama_modules.utils import show3D
    print(PositionalEncoding_3D_sincos(6*64, [50, 60, 70]).shape)
    pe = tensor2array(torch.squeeze(PositionalEncoding_3D_sincos(6*64, [50, 60, 70])))
    show3D(pe[:,:,:,35])
    show3D(pe[:,:,30,:])
    show3D(pe[:,25,:,:])

    pe = tensor2array(torch.squeeze(PositionalEncoding_3D_sincos(6*64, [50, 60, 70], temperature=100)))
    show3D(pe[:,:,:,35])
    show3D(pe[:,:,30,:])
    show3D(pe[:,25,:,:])


    pe = tensor2array(torch.squeeze(PositionalEncoding_3D_sincos(6*64, [50, 60, 70], temperature=100, return_flatten=True).reshape(1,6*64,50,60,70)
    ))
    show3D(pe[:,:,:,35])
    show3D(pe[:,:,30,:])
    show3D(pe[:,25,:,:])
    """
    if embedding_dim % 6 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(embedding_dim))
    height, width, length = token_shape
    pe = torch.zeros(embedding_dim, height, width, length)
    # Each dimension use 1/3 of d_model
    or_embedding_dim = embedding_dim
    embedding_dim = int(embedding_dim / 3)
    div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(temperature) / embedding_dim))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_l = torch.arange(0., length).unsqueeze(1)
    pe[0:embedding_dim:2, :, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1,length)
    pe[1:embedding_dim:2, :, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1,length)
    pe[embedding_dim:embedding_dim*2:2, :, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, width,length)
    pe[embedding_dim + 1:embedding_dim*2+1:2, :, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, width,length)
    pe[embedding_dim*2::2, :, :, :] = torch.sin(pos_l * div_term).transpose(0, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, height, width, 1)
    pe[embedding_dim*2+1::2, :, :, :] = torch.cos(pos_l * div_term).transpose(0, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, height, width, 1)
    pe = pe.unsqueeze(0)

    if return_flatten:
        return pe.reshape(1, token_shape[0]*token_shape[1]*token_shape[2], or_embedding_dim)
    else:
        return pe

