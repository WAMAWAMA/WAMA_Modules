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
    :param norm: bn(batch) or gn(group) or in(instance) or ln(layer)
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


def MakeActive(active='relu'):
    """
    :param active: relu or leakyrelu
    :return:
    """
    if active == 'relu':
        return nn.ReLU(inplace=True)
    elif active == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
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
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, norm='bn', active='relu', gn_c = 8, dim = 2):
        """
        # convolution + normalization + activation
        :param in_channels:
        :param out_channels:
        :param stride:
        :param norm: bn(batch) or gn(group) or in(instance) or ln(layer)
        :param active: relu leakyrelu
        :param gn_c: coordinate groups of gn
        """
        super().__init__()

        self.conv = MakeConv(in_channels, out_channels, kernel_size, padding=1, stride=stride, dim = dim)
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































# vgg 系列
class VGGBlock(nn.Module):
    """
    简洁的CBA结构（Conv，bn，activation）
    输入输出形状保持一致
    """
    def __init__(self, in_channels, middle_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels  =in_channels
        self.out_channels  =out_channels

        # a
        self.relu = nn.ReLU(inplace=True)

        # cb1
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, stride=stride)

        # print(norm_set)
        if norm_set == 'bn':
            self.bn1 = nn.BatchNorm2d(middle_channels)
        elif norm_set == 'gn':
            self.bn1 = nn.GroupNorm(gn_c, middle_channels)
        elif norm_set == 'in':
            self.bn1 = nn.InstanceNorm2d(middle_channels)

        # cb2
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, stride=stride)
        if norm_set == 'bn':
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm_set == 'gn':
            self.bn2 = nn.GroupNorm(gn_c, out_channels)
        elif norm_set == 'in':
            self.bn2 = nn.InstanceNorm2d(out_channels)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class VGGBlock3D(nn.Module):
    """
    简洁的CBA结构（Conv，bn，activation）
    输入输出形状保持一致
    """
    def __init__(self, in_channels, middle_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels  =in_channels
        self.out_channels  =out_channels



        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1, stride=stride)

        # print(norm_set)
        if norm_set == 'bn':
            self.bn1 = nn.BatchNorm3d(middle_channels)
        elif norm_set == 'gn':
            self.bn1 = nn.GroupNorm(gn_c, middle_channels)
        elif norm_set == 'in':
            self.bn1 = nn.InstanceNorm3d(middle_channels)




        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1, stride=stride)
        if norm_set == 'bn':
            self.bn2 = nn.BatchNorm3d(out_channels)
        elif norm_set == 'gn':
            self.bn2 = nn.GroupNorm(gn_c, out_channels)
        elif norm_set == 'in':
            self.bn2 = nn.InstanceNorm3d(out_channels)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


# resnet系列
class ResBlock(nn.Module):
    """
    resnet基础block,2D和3D兼容
    包含3个 cbn结构，如果in_channels和out_channels不一致，则增加一个projection（也是cbn）来把输入x映射到out_channels

    # 2D demo
    input = torch.ones([3,128,64,64]).cuda()
    resblock1 = ResBlock(in_channels=128,middle_channels=256,out_channels=512).cuda()
    output = resblock1(input)
    # 3D demo
    input = torch.ones([3,128,32,32,32]).cuda()
    resblock1 = ResBlock(in_channels=128,middle_channels=256,out_channels=512, dim=3).cuda()
    output = resblock1(input)

    """
    def __init__(self, in_channels, middle_channels, out_channels, stride=1, dim = 2):
        super().__init__()
        self.in_channels  =in_channels
        self.middle_channels  =middle_channels
        self.out_channels  =out_channels
        self.dim = dim

        if self.dim ==2:
            make_conv = ConvNormActive2D
        elif dim ==3:
            make_conv = ConvNormActive3D
        else:
            raise ValueError('dim should be 2 or 3')

        # 构建projection
        if self.out_channels != self.in_channels: # 输入输出channel不一样，则添加projection改变输入通道,用于不同stage的过渡处
            self.projection = make_conv(in_channels=self.in_channels,out_channels=self.out_channels)
        else:
            self.projection = None

        self.conv1 = make_conv(self.in_channels, self.middle_channels)
        self.conv2 = make_conv(self.middle_channels, self.middle_channels)
        self.conv3 = make_conv(self.middle_channels, self.out_channels)

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        if self.projection is not None:
            return f+self.projection(x)
        else:
            return f+x

class ResStage(nn.Module):
    """
    一个ResStage包括n个ResBlock
    ResStage 和 VGGBlock 级别一样
    第一个resblock的channel会不一样（因为要负责改变channel）
    其余block结构都一样，输入输出的shape和channel都一样

    # 2D demo
    input = torch.ones([3,32,64,64]).cuda()
    resblock1 = ResStage(in_channels=32,middle_channels=128,out_channels=256, dim=2, block_num=3).cuda()
    output = resblock1(input)
    # 3D demo
    input = torch.ones([3,32,32,32,32]).cuda()
    resblock1 = ResStage(in_channels=32,middle_channels=128,out_channels=256, dim=3, block_num=3).cuda()
    output = resblock1(input)
    """
    def __init__(self, in_channels, middle_channels, out_channels, block_num=3, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.block_num = block_num
        self.dim = dim

        print('stage contains ',block_num, ' blocks')

        # 构建block
        self.resblock_list = nn.ModuleList([])
        for index in range(self.block_num):
            if index == 0:
                self.resblock_list.append(ResBlock(in_channels, middle_channels, out_channels, dim=dim))
            else:
                self.resblock_list.append(ResBlock(out_channels, middle_channels, out_channels, dim=dim))

    def forward(self, x):
        for resblock in self.resblock_list:
            x = resblock(x)
        return x







