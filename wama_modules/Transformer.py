"""
Part of the code
from https://github.com/lucidrains/vit-pytorch
from https://github.com/SlongLiu/query2labels
from https://github.com/facebookresearch/detr
Thanks a lot to the authors of these repos
"""


import torch
from torch import nn, einsum
from einops import rearrange


class FeedForward(nn.Module):
    """
    Just Is MLP, the input and output shape is consistent
    hidden_dim: The intermediate dimension of feature extraction
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # could be relu
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,):
        """
        :param dim: input tokens channel, input token size = [bz, token_num, channel]
        :param heads:
        :param dim_head: projection dim of qkv
        :param dropout:
        """
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

    def forward(self, q_tokens, k_tokens=None, v_tokens=None):
        """
        # demo1: self attention
        tokens = torch.ones([3, 5, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim = 512
        attention_layer = MultiHeadAttention(dim, heads, dim_head)
        tokens_, attention_maps = attention_layer(tokens)
        print(tokens.shape, tokens_.shape, attention_maps.shape)


        # demo2: cross attention
        q_tokens = torch.ones([3, 5, 512])
        k_tokens = torch.ones([3, 15, 512])
        v_tokens = torch.ones([3, 15, 512]) # the same size as k_tokens
        dim = 512
        heads = 8
        dim_head = 32
        dim = 512
        attention_layer = MultiHeadAttention(dim, heads, dim_head)
        tokens_, attention_maps = attention_layer(q_tokens, k_tokens, v_tokens)
        print(tokens.shape, tokens_.shape, attention_maps.shape)

        """

        if k_tokens is None and v_tokens is None:
            v_tokens = k_tokens = q_tokens
        elif k_tokens is not None and v_tokens is not None:
            pass
        else:
            raise ValueError('k_tokens and v_tokens should be None or not simultaneously')

        q, k, v = [
            self.to_q(q_tokens),
            self.to_k(k_tokens),
            self.to_v(v_tokens),
        ]

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, token_channels, heads=8, dim_head=64, channel_mlp=None, dropout=0., AddPosEmb2Value=True):
        """
        :param dim: input tokens channel, input token size = [bz, token_num, channel]
        :param heads:
        :param dim_head: projection dim of qkv
        :param dropout:

        like what Vit encoder does, AddPosEmb2Value = True
        Vit,An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale(https://arxiv.org/abs/2010.11929)

        """
        super().__init__()
        if channel_mlp is None:
            channel_mlp = token_channels
        self.AddPosEmb2Value = AddPosEmb2Value

        self.norm1 = nn.LayerNorm(token_channels)
        self.Attention = MultiHeadAttention(token_channels, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(token_channels)
        self.FFN = FeedForward(token_channels, channel_mlp, dropout=dropout)

    def forward(self, tokens, pos_embeddings=None):
        """
        # demo1: no position_embedding --------------------------------------------
        tokens = torch.ones([3, 5, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerEncoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=True)
        tokens_, attention_maps = attention_layer(tokens)
        print(tokens.shape, tokens_.shape, attention_maps.shape)

        # demo2: qkv with position_embedding --------------------------------------------
        tokens = torch.ones([3, 5, 512])
        position_embedding = torch.ones([3, 5, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerEncoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=True)
        tokens_, attention_maps = attention_layer(tokens, position_embedding) # way1 to add position_embedding
        tokens_, attention_maps = attention_layer(tokens+position_embedding) # way2 to add position_embedding
        print(tokens.shape, tokens_.shape, attention_maps.shape)

        # demo3: qk with position_embedding, but v without position_embedding------------
        tokens = torch.ones([3, 5, 512])
        position_embedding = torch.ones([3, 5, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerEncoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=False)
        tokens_, attention_maps = attention_layer(tokens, position_embedding)
        print(tokens.shape, tokens_.shape, attention_maps.shape)

        """

        # pre norm
        q = k = v = self.norm1(tokens)

        # add pos_embeddings
        if pos_embeddings is not None:
            print('add pos_embeddings to q and k')
            q = q + pos_embeddings
            k = k + pos_embeddings
            if self.AddPosEmb2Value:
                print('add pos_embeddings to v')
                v = v + pos_embeddings

        # self attention
        attn_q, attn_map = self.Attention(q, k, v)  # attn_q has same shape with q

        # residual
        tokens = tokens + attn_q  # todo add token (which is not normed), not v !

        # norm
        q = self.norm2(tokens)

        # FFN
        ffn_q = self.FFN(q)

        # residual
        tokens = tokens + ffn_q  # todo add token (which is not normed), not v !

        return tokens, attn_map


class TransformerDecoderLayer(nn.Module):
    def __init__(self, token_channels, heads=8, dim_head=64, channel_mlp=None, dropout=0., AddPosEmb2Value=False):
        """
        :param dim: input tokens channel, input token size = [bz, token_num, channel]
        :param heads:
        :param dim_head: projection dim of qkv
        :param dropout:

        like what DETR/Q2L decoder does, 'AddPosEmb2Value' is False
        DETR,End-to-End Object Detection with Transformers(https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13)
        Q2L,Query2Label: A Simple Transformer Way to Multi-Label Classification(https://arxiv.org/abs/2107.10834)
        """

        super().__init__()
        if channel_mlp is None:
            channel_mlp = token_channels
        self.AddPosEmb2Value = AddPosEmb2Value

        # stage1 self attention
        self.norm1 = nn.LayerNorm(token_channels)
        self.SelfAttention = MultiHeadAttention(token_channels, heads, dim_head, dropout)

        # stage2 cross attention
        self.norm2_q = nn.LayerNorm(token_channels)
        self.norm2_k = nn.LayerNorm(token_channels)
        self.CrossAttention = MultiHeadAttention(token_channels, heads, dim_head, dropout)

        # ffn
        self.norm3 = nn.LayerNorm(token_channels)
        self.FFN = FeedForward(token_channels, channel_mlp, dropout=dropout)

    def forward(self, q_tokens, v_tokens, q_pos_embeddings=None, v_pos_embeddings=None):
        """
        q_tokens = [bz, token_num1, channel]
        v_tokens = [bz, token_num2, channel]
        q_tokens and v_tokens have same channel

        # demo1: no position_embedding --------------------------------------------
        q_tokens = torch.ones([3, 5, 512])
        v_tokens = torch.ones([3, 15, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerDecoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=True)
        q_tokens_, self_attention_maps, cross_attention_maps = attention_layer(q_tokens, v_tokens)
        print(q_tokens.shape, tokens_.shape)
        print(self_attention_maps.shape, cross_attention_maps.shape)


        # demo2: qkv with position_embedding --------------------------------------------
        q_tokens = torch.ones([3, 5, 512])
        q_posemb = torch.ones([3, 5, 512])
        v_tokens = torch.ones([3, 15, 512])
        v_posemb = torch.ones([3, 15, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerDecoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=True)
        q_tokens_, self_attention_maps, cross_attention_maps = attention_layer(q_tokens, v_tokens, q_posemb, v_posemb) # way 1 to add posemb
        q_tokens_, self_attention_maps, cross_attention_maps = attention_layer(q_tokens+q_posemb, v_tokens+v_posemb) # way 2 to add posemb
        print(q_tokens.shape, tokens_.shape)
        print(self_attention_maps.shape, cross_attention_maps.shape)



        # demo3 (same as DETR): qk with position_embedding, but v without position_embedding------------
        q_tokens = torch.ones([3, 5, 512])
        q_posemb = torch.ones([3, 5, 512])
        v_tokens = torch.ones([3, 15, 512])
        v_posemb = torch.ones([3, 15, 512])
        dim = 512
        heads = 8
        dim_head = 32
        dim_mlp = 64
        attention_layer = TransformerDecoderLayer(dim, heads, dim_head, dim_mlp, AddPosEmb2Value=False)
        q_tokens_, self_attention_maps, cross_attention_maps = attention_layer(q_tokens, v_tokens, q_posemb, v_posemb)
        print(q_tokens.shape, tokens_.shape)
        print(self_attention_maps.shape, cross_attention_maps.shape)

        """

        # todo stage1: self attention with q_tokens -------------------------------
        if True:
            # pre norm
            q = k = v = self.norm1(q_tokens)

            # add pos_embeddings
            if q_pos_embeddings is not None:
                print('self attention: add pos_embeddings to q and k')
                q = q + q_pos_embeddings
                k = k + q_pos_embeddings
                if self.AddPosEmb2Value:
                    print('self attention: add pos_embeddings to v')
                    v = v + q_pos_embeddings

            # self attention
            attn_q, self_attn_map = self.SelfAttention(q, k, v)  # attn_q has same shape with q

            # residual
            q_tokens = q_tokens + attn_q

        # todo stage2: cross attention with q_tokens -------------------------------
        if True:
            # norm q and add q pos_embeddings
            q = self.norm2_q(q_tokens)
            if q_pos_embeddings is not None:
                print('cross attention: add pos_embeddings to q')
                q = q + q_pos_embeddings

            # norm kv and add kv pos_embeddings
            k = self.norm2_k(v_tokens)  # todo: add, different from official code
            v = self.norm2_k(v_tokens)  # todo: add, different from official code
            if v_pos_embeddings is not None:
                print('cross attention: add pos_embeddings to k')
                k = k + v_pos_embeddings
                if self.AddPosEmb2Value:
                    print('cross attention: add pos_embeddings to v')
                    v = v + v_pos_embeddings

            # cross attention
            attn_q, cross_attn_map = self.CrossAttention(q, k, v)  # attn_q has same shape with q

            # residual
            q_tokens = q_tokens + attn_q

        # todo stage3:FFN ------------------------------------------------
        if True:
            # norm
            q = self.norm3(q_tokens)

            # FFN
            ffn_q = self.FFN(q)

            # residual
            q_tokens = q_tokens + ffn_q

        return q_tokens, self_attn_map, cross_attn_map







