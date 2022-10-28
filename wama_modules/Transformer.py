"""
from https://github.com/lucidrains/vit-pytorch
Thanks a lot to the author https://github.com/lucidrains
"""


import torch
from torch import nn, einsum
from einops import rearrange
from wama_modules.utils import tmp_class


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Just Is MLP, the input and output shape is consistent
    hidden_dim: The intermediate dimension of feature extraction
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """
    totally self_attention
        dim_head: In order to map the input X to QKV, it is necessary to multiply with the W matrix.
        This process is implemented with a full connection.
        dim_head is the output shape of specified W (also the shape of QKV)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # print(attn.shape)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class CrossAttention(nn.Module):
    """
        dim_head: In order to map the input X to QKV, it is necessary to multiply with the W matrix.
        This process is implemented with a full connection.
        dim_head is the output shape of specified W (also the shape of QKV)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q_tokens, kv_tokens):
        """
        q_tokens = torch.ones([3, 5, 512])
        kv_tokens = torch.ones([3, 20, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim = 512
        self = tmp_class()
        """
        b_q, n_q, _, h_q = *q_tokens.shape, self.heads
        b_kv, n_kv, _, h_kv = *kv_tokens.shape, self.heads
        q, k, v = [
            self.to_q(q_tokens),
            self.to_k(kv_tokens),
            self.to_v(kv_tokens),
        ]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h_q)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h_kv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h_kv)

        # _ = [print(i.shape) for i in [q,k,v]]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # print(attn.shape)
        # print(v.shape)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer_encoder(nn.Module):
    """
    attention is all you need
    the Transformer is just a encoder
    """
    def __init__(self, channel, layer_number, head_number, channel_head, channel_mlp, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layer_number):
            self.layers.append(nn.ModuleList([
                PreNorm(channel, SelfAttention(channel, heads=head_number, dim_head=channel_head, dropout=dropout)),
                PreNorm(channel, FeedForward(channel, channel_mlp, dropout=dropout))
            ]))

    def forward(self, x):
        """
        :param x: [bz, token_number, channel]
        :return:

        tokens = torch.ones([3, 15, 512])
        trans = Transformer_encoder(channel=512, layer_number=3, head_number=8, channel_head=64, channel_mlp=128, dropout=0.)
        tokens_out, attn_map_list = trans(tokens)

        """
        attn_map_list = []  # from layer first to last
        for attn, ff in self.layers:
            attn_f, attn_map = attn(x)
            attn_map_list.append(attn_map)
            x = attn_f + x
            x = ff(x) + x
        return x, attn_map_list
