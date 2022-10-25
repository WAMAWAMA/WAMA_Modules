import torch.nn as nn
import torch
import torch.nn.functional as F
from wama_modules.utils import tensor2array


class GlobalAvgPool(nn.Module):
    """Global average pooling over the input's spatial dimensions"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
        inputs = torch.ones([3,12])
        inputs = torch.ones([3,12,13]) # 1D
        inputs = torch.ones([3,12,13,13]) # 2D
        inputs = torch.ones([3,12,13,13,13]) # 3D
        """
        if len(inputs.shape) == 2:
            return inputs
        elif len(inputs.shape) == 3:
            return nn.functional.adaptive_avg_pool1d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 4:
            return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 5:
            return nn.functional.adaptive_avg_pool3d(inputs, 1).view(inputs.size(0), -1)


class GlobalMaxPool(nn.Module):
    def __init__(self):
        """Global max pooling over the input's spatial dimensions"""
        super().__init__()
    def forward(self, inputs):
        """
        inputs = torch.ones([3,12])
        inputs = torch.ones([3,12,13]) # 1D
        inputs = torch.ones([3,12,13,13]) # 2D
        inputs = torch.ones([3,12,13,13,13]) # 3D
        """
        if len(inputs.shape) == 2:
            return inputs
        elif len(inputs.shape) == 3:
            return nn.functional.adaptive_max_pool1d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 4:
            return nn.functional.adaptive_max_pool2d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 5:
            return nn.functional.adaptive_max_pool3d(inputs, 1).view(inputs.size(0), -1)


class GlobalMaxAvgPool(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super().__init__()
        self.GAP = GlobalAvgPool()
        self.GMP = GlobalMaxPool()
    def forward(self, inputs):
        """
        demo:
        inputs = torch.randn([2,1,2]) # 1D
        inputs = torch.randn([2,1,2,2]) # 2D
        inputs = torch.randn([2,1,2,2,2]) # 3D
        print(inputs.shape)
        print(tensor2array(inputs))
        gmap = GlobalMaxAvgPool()
        gap = GlobalAvgPool()
        gmp = GlobalMaxPool()
        outputs_gmap = gmap(inputs)
        outputs_gap = gap(inputs)
        outputs_gmp = gmp(inputs)
        print(outputs_gmap.shape)
        print(outputs_gmap) # should be equal
        print(outputs_gap*0.5+outputs_gmp*0.5) # should be equal with outputs_gmap
        """
        return (self.GMP(inputs) + self.GAP(inputs))/2.


def customLayerNorm(x, esp = 1e-6):
    """
    :param x: [bz, c, **shape] 1D/2D/3D
    :return:

    # demo

    x = torch.randn([2,2,3])*10 # 1D
    ln1 = customLayerNorm
    ln2 = nn.LayerNorm([3], eps=1e-6)
    y1 = ln1(x)
    y2 = ln2(x)
    print(y1)
    print(y2)

    x = torch.randn([2,2,1,3])*10 # 2D
    ln1 = customLayerNorm
    ln2 = nn.LayerNorm([1,3], eps=1e-6)
    y1 = ln1(x)
    y2 = ln2(x)
    print(y1)
    print(y2)

    x = torch.randn([2,2,1,3,1])*10 # 3D
    ln1 = customLayerNorm
    ln2 = nn.LayerNorm([1,3,1], eps=1e-6)
    y1 = ln1(x)
    y2 = ln2(x)
    print(y1)
    print(y2)

    """

    mean = torch.mean(x, [i+2 for i in range(len(x.shape)-2)])
    var = torch.var(x, [i+2 for i in range(len(x.shape)-2)], False)
    for _ in range(len(x.shape)-2):
        mean = torch.unsqueeze(mean, -1)
        var = torch.unsqueeze(var, -1)
    y = (x-mean) / (torch.sqrt(var)+esp)
    return y


def MakeNorm(dim, channel, norm='bn', gn_c = 8):
    """
    :param dim: input dimetions, 1D/2D/3D
    :param norm: bn(batch) or gn(group) or in(instance) or ln(layer) or None(identity mapping)
    :return:
    """
    if norm == 'bn':
        if dim == 1:
            return nn.BatchNorm1d(channel)
        elif dim == 2:
            return nn.BatchNorm2d(channel)
        elif dim == 3:
            return nn.BatchNorm3d(channel)
    elif norm == 'in':
        if dim == 1:
            return nn.InstanceNorm1d(channel)
        elif dim == 2:
            return nn.InstanceNorm2d(channel)
        elif dim == 3:
            return nn.InstanceNorm3d(channel)
    elif norm == 'gn':
        return nn.GroupNorm(gn_c, channel)
    elif norm == 'ln':
        return customLayerNorm
    elif norm == 'None' or norm is None:
        return nn.Identity()


def MakeActive(active='relu'):
    """
    :param active: relu or leakyrelu or None(identity mapping)
    :return:
    """
    if active == 'relu':
        return nn.ReLU(inplace=True)
    elif active == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif active == 'None' or active is None:
        return nn.Identity()
    else:
        raise ValueError('should be relu or leakyrelu')


def MakeConv(in_channels, out_channels, kernel_size, padding=1, stride=1, dim = 2):
    if dim == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
    elif dim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
    elif dim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)


class ConvNormActive(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, norm='bn', active='relu', gn_c = 8, dim = 2, padding = 1):
        """
        # convolution + normalization + activation
        :param in_channels:
        :param out_channels:
        :param stride:
        :param kernel_size:
        :param norm: bn(batch) or gn(group) or in(instance) or ln(layer) or None(identity mapping)
        :param active: relu or leakyrelu or None(identity mapping)
        :param gn_c: coordinate groups of gn
        :param dim: 1\2\3D network
        """
        super().__init__()

        self.conv = MakeConv(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dim = dim)
        self.norm = MakeNorm(dim, out_channels, norm, gn_c)
        self.active = MakeActive(active)

    def forward(self, x):
        """
        # demo
        x = torch.randn([12,16,2]) # 1D
        dim = 1
        layer = ConvNormActive(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2]) # 2D
        dim = 2
        layer = ConvNormActive(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2,2]) # 2D
        dim = 3
        layer = ConvNormActive(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
        layer = ConvNormActive(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.active(out)
        return out


class VGGBlock(ConvNormActive):
    """
    is ConvNormActive，but kernel size = 3x3

    # demo
    x = torch.randn([12,16,2]) # 1D
    dim = 1
    layer = VGGBlock(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'None', 'None', 8, dim = dim)
    y = layer(x)
    print(x.shape)
    print(y.shape)

    x = torch.randn([12,16,2,2]) # 2D
    dim = 2
    layer = VGGBlock(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
    y = layer(x)
    print(x.shape)
    print(y.shape)

    x = torch.randn([12,16,2,2,2]) # 2D
    dim = 3
    layer = VGGBlock(16, 32, 1, 3, 'gn', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'ln', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'in', 'relu', 8, dim = dim)
    layer = VGGBlock(16, 32, 1, 3, 'bn', 'relu', 8, dim = dim)
    y = layer(x)
    print(x.shape)
    print(y.shape)

    """
    def __init__(self, *args, **kwarg):
        """
        :param in_channels:
        :param out_channels:
        :param stride:
        :param kernel_size:
        :param norm: bn(batch) or gn(group) or in(instance) or ln(layer) or None(identity mapping)
        :param active: relu or leakyrelu or None(identity mapping)
        :param gn_c: coordinate groups of gn
        :param dim: 1\2\3D network
        """
        super().__init__(*args, **kwarg)


class VGGStage(nn.Module):
    """
    a VGGStage contains multiple VGGBlocks

    """
    def __init__(self, in_channels, out_channels, block_num=2, norm='bn', active='relu', gn_c=8, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_num = block_num
        self.dim = dim

        print('VGGStage stage contains ', block_num, ' blocks')

        # 构建block
        self.block_list = nn.ModuleList([])
        for index in range(self.block_num):
            if index == 0:
                self.block_list.append(VGGBlock(in_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))
            else:
                self.block_list.append(VGGBlock(out_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))

    def forward(self, x):
        """
        # demo
        x = torch.randn([12,16,2]) # 1D
        dim = 1
        block_num = 3
        layer = VGGStage(16, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'ln', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'in', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2]) # 2D
        dim = 2
        block_num = 3
        layer = VGGStage(16, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'ln', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'in', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2,2]) # 3D
        dim = 3
        block_num = 3
        layer = VGGStage(16, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'ln', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'in', 'relu', 8, dim = dim)
        layer = VGGStage(16, 32, block_num, 'bn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        """
        for block in self.block_list:
            x = block(x)
        return x


class ResBlock(nn.Module):
    """
    there two types of ResBlock

    type1: '33', used for ResNet18 and ResNet34
    x_in → conv3x3 → norm → active → conv3x3 → norm → active → x_out
        ↘--→ conv1x1 if in_c != out_c) ---↗

    type1: '131', used for ResNet50 and ResNet101 and ResNet152
    x_in → conv3x3 → norm → active → conv3x3 → norm → active → x_out
        ↘--→ conv1x1 if in_c != out_c) ---↗
    """
    def __init__(self, type, in_channels, middle_channels, out_channels, norm='bn', active='relu', gn_c=8, dim=2):
        super().__init__()
        self.type = type
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.dim = dim

        """
        If the input/output channels are different, 
        add projection to alter the input channel for transitions at different stages
        """
        if self.out_channels != self.in_channels:
            self.projection = MakeConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, dim=dim,padding=0)
        else:
            self.projection = nn.Identity()


        # build list
        if self.type == '33':
            self.conv_list = nn.ModuleList([])
            self.conv_list.append(ConvNormActive(in_channels, out_channels, kernel_size=3, norm=norm, active=active, gn_c = gn_c, dim = dim))
            self.conv_list.append(ConvNormActive(out_channels, out_channels, kernel_size=3, norm=norm, active='None', gn_c = gn_c, dim = dim))
        elif self.type == '131':
            self.conv_list = nn.ModuleList([])
            self.conv_list.append(ConvNormActive(in_channels, middle_channels, kernel_size=1, norm=norm, active=active, gn_c = gn_c, dim = dim, padding=0))
            self.conv_list.append(ConvNormActive(middle_channels, middle_channels, kernel_size=3, norm=norm, active=active, gn_c = gn_c, dim = dim))
            self.conv_list.append(ConvNormActive(middle_channels, out_channels, kernel_size=1, norm=norm, active='None', gn_c = gn_c, dim = dim, padding=0))
        else:
            raise ValueError('type of ResBlock must be 131 or 33,'+type+' is not acceptable')

        # out activation
        self.activation = MakeActive(active)

    def forward(self, x):
        """
        # demo
        x = torch.randn([12,16,2]) # 1D
        dim = 1
        layer = ResBlock('131', 16, 64, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('131', 16, 64, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2]) # 2D
        dim = 2
        layer = ResBlock('131', 16, 64, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('131', 16, 64, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2,2]) # 3D
        dim = 3
        layer = ResBlock('131', 16, 64, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 32, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('131', 16, 64, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        layer = ResBlock('33', 16, None, 16, norm='bn', active='relu', gn_c=8, dim=dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        """
        for conv_index, conv in enumerate(self.conv_list):
            if conv_index == 0:
                f = conv(x)
            else:
                f = conv(f)

        return self.activation(f + self.projection(x))


class ResStage(nn.Module):
    """
    a ResStage contains multiple ResBlocks

    """
    def __init__(self, type, in_channels, middle_channels, out_channels, block_num=2, norm='bn', active='relu', gn_c=8, dim=2):
        super().__init__()
        self.type = type
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.block_num = block_num
        self.dim = dim

        print('ResStage stage contains ', block_num, ' blocks')

        # 构建block
        if type == '33':
            self.block_list = nn.ModuleList([])
            for index in range(self.block_num):
                if index == 0:
                    self.block_list.append(ResBlock(type, in_channels, None, out_channels, norm=norm, active=active, gn_c=gn_c, dim=dim))
                else:
                    self.block_list.append(ResBlock(type, out_channels, None, out_channels, norm=norm, active=active, gn_c=gn_c, dim=dim))
        elif type == '131':
            self.block_list = nn.ModuleList([])
            for index in range(self.block_num):
                if index == 0:
                    self.block_list.append(ResBlock(type, in_channels, middle_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))
                else:
                    self.block_list.append(ResBlock(type, out_channels, middle_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))

    def forward(self, x):
        """
        # demo
        x = torch.randn([12,16,2]) # 1D
        dim = 1
        block_num = 3
        layer = ResStage('33', 16, None, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = ResStage('131', 16, 32, 64, block_num, 'gn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2]) # 2D
        dim = 2
        block_num = 3
        layer = ResStage('33', 16, None, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = ResStage('131', 16, 32, 64, block_num, 'gn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)

        x = torch.randn([12,16,2,2,2]) # 3D
        dim = 3
        block_num = 3
        layer = ResStage('33', 16, None, 32, block_num, 'gn', 'relu', 8, dim = dim)
        layer = ResStage('131', 16, 32, 64, block_num, 'gn', 'relu', 8, dim = dim)
        y = layer(x)
        print(x.shape)
        print(y.shape)
        """
        for block in self.block_list:
            x = block(x)
        return x








