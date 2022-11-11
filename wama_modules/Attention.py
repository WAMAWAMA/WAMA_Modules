import torch.nn as nn
import torch
import torch.nn.functional as F
from wama_modules.BaseModule import MakeNorm
from wama_modules.utils import tmp_class, tensor2array


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=None, which_first='c', dim=2):
        """
        :param in_channels:
        :param reduction:
        :param which_first: c or s, refer to 'perform channel attention or spatial attention first'  None perform c and s parallel
        :param dim: 1/2/3
        """
        super().__init__()
        self.which_first = which_first
        self.dim = dim

        if reduction is None:
            reduction = in_channels

        if in_channels % reduction != 0:
            raise ValueError('in_channels % reduction should be 0')

        if self.dim == 1:
            make_conv = nn.Conv1d
            pooling = nn.AdaptiveAvgPool1d
        elif self.dim == 2:
            make_conv = nn.Conv2d
            pooling = nn.AdaptiveAvgPool2d
        elif self.dim == 3:
            make_conv = nn.Conv3d
            pooling = nn.AdaptiveAvgPool3d

        self.cSE = nn.Sequential(
            pooling(1),
            make_conv(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            make_conv(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            make_conv(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        :param x: 1D/2D/3D is ok
        :return:

        # demo
        x = torch.ones([3,12,4]) # 1D
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='c', dim=1)
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='s', dim=1)
        attention_layer = SCSEModule(in_channels=12, reduction=3, which_first='s', dim=1)
        x1 = attention_layer(x)

        x = torch.ones([3,12,4,3]) # 2D
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='c', dim=2)
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='s', dim=2)
        attention_layer = SCSEModule(in_channels=12, reduction=3, which_first='s', dim=2)
        x1 = attention_layer(x)

        x = torch.ones([3,12,4,3,3]) # 3D
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='c', dim=3)
        attention_layer = SCSEModule(in_channels=12, reduction=None, which_first='s', dim=3)
        attention_layer = SCSEModule(in_channels=12, reduction=3, which_first='s', dim=3)
        x1 = attention_layer(x)

        """
        if self.which_first == None:
            return x * self.cSE(x) + x * self.sSE(x)
        elif self.which_first == 'c':
            x = x * self.cSE(x)
            return x * self.sSE(x)
        elif self.which_first == 's':
            x = x * self.sSE(x)
            return x * self.cSE(x)


class NonLocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None, use_bn = False, dim=2):
        """
        NonLocal with 'embedded' mode
        :param in_channels:
        :param reduction:
        :param which_first: c or s, refer to 'perform channel attention or spatial attention first'  None perform c and s parallel
        :param dim: 1/2/3
        """
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.use_bn = use_bn
        self.dim = dim


        # recommended: the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        if self.dim == 1:
            make_conv = nn.Conv1d
        elif self.dim == 2:
            make_conv = nn.Conv2d
        elif self.dim == 3:
            make_conv = nn.Conv3d

        # layers θ Φ g
        self.theta = make_conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)  # θ
        self.phi = make_conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)  # Φ
        self.g = make_conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)  # g


        #  the last conv layer
        if self.use_bn:
            self.final_conv = nn.Sequential(
                    make_conv(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    MakeNorm(dim, self.in_channels, norm='bn', gn_c=8)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.final_conv[1].weight, 0)
            nn.init.constant_(self.final_conv[1].bias, 0)
        else:
            self.final_conv = make_conv(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.final_conv.weight, 0)
            nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x):
        """

        # demo
        x = torch.randn([2,12,16])  # 1D
        attention_layer = NonLocal(12, inter_channels=None, use_bn = False, dim=1)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = False, dim=1)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = True, dim=1)
        x1 = attention_layer(x)

        x = torch.randn([2,12,16,16])  # 2D
        attention_layer = NonLocal(12, inter_channels=None, use_bn = False, dim=2)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = False, dim=2)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = True, dim=2)
        x1 = attention_layer(x)

        x = torch.randn([2,12,16,16,16])  # 3D
        attention_layer = NonLocal(12, inter_channels=None, use_bn = False, dim=3)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = False, dim=3)
        attention_layer = NonLocal(12, inter_channels=6, use_bn = True, dim=3)
        x1 = attention_layer(x)

        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1) # [bz, *shape, channel]

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # Φ  [bz, channel, *shape]
        theta_x = theta_x.permute(0, 2, 1) # θ [bz, *shape, channel]
        f = torch.matmul(theta_x, phi_x) # [bz, *shape, channel] *  [bz, channel, *shape] =  [bz, *shape, *shape]
        f_div_C = F.softmax(f, dim=-1) # [bz, *shape, *shape] f_div_C[0].sum(-1) is all 1, because every row is an attention vector
        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.final_conv(y)

        # residual connection
        z = W_y + x

        return z


# STN