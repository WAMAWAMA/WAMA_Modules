from wama_modules.BaseModule import *
from wama_modules.utils import resizeTensor


# vgg
class VGGEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 stage_output_channels=[64, 128, 256, 512],
                 blocks=[6, 12, 24, 16],
                 downsample_ration=[0.5, 0.5, 0.5, 0.5],
                 downsample_first=False,
                 norm='bn',
                 active='relu',
                 gn_c=8,
                 dim=2):
        super().__init__()

        self.dim = dim
        self.blocks = blocks
        self.downsample_ration =downsample_ration
        self.downsample_first =downsample_first

        # res stages
        self.vgg_stage = nn.ModuleList([])
        for stage_index in range(len(stage_output_channels)):
            if stage_index == 0:
                self.vgg_stage.append(VGGStage(in_channels, stage_output_channels[stage_index], block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))
            else:
                self.vgg_stage.append(VGGStage(stage_output_channels[stage_index-1], stage_output_channels[stage_index], block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))

    def forward(self, x):
        """
        # demo 1D
        dim = 1
        in_channels = 3
        input = torch.ones([2,in_channels, 128])
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]

        # demo 2D
        dim = 2
        in_channels = 3
        input = torch.ones([2,in_channels, 128, 128])
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]

        # demo 3D
        dim = 3
        in_channels = 3
        input = torch.ones([2,in_channels, 64, 64, 64])
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        encoder = VGGEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[2, 2, 2], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]

        """

        f = x
        stage_features = []
        for stage_index, downsample_ratio in enumerate(self.downsample_ration):
            if self.downsample_first:
                f = resizeTensor(f, scale_factor=downsample_ratio)
                f = self.vgg_stage[stage_index](f)
            else:
                f = self.vgg_stage[stage_index](f)
                f = resizeTensor(f, scale_factor=downsample_ratio)
            stage_features.append(f)

        return stage_features


def VGG16Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = VGG16Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = VGG16Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 3D
    dim = 3
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128, 128])
    encoder = VGG16Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return VGGEncoder(in_channels,
                      stage_output_channels=[64, 128, 256, 512, 512],
                      blocks=[2, 2, 3, 3, 3],
                      downsample_ration=[0.5, 0.5, 0.5, 0.5, 0.5],
                      dim=dim)


def VGG19Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = VGG19Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = VGG19Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 3D
    dim = 3
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128, 128])
    encoder = VGG19Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return VGGEncoder(in_channels,
                      stage_output_channels=[64, 128, 256, 512, 512],
                      blocks=[2, 2, 4, 4, 4],
                      downsample_ration=[0.5, 0.5, 0.5, 0.5, 0.5],
                      dim=dim)


# resnet
class ResNetEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 stage_output_channels=[64, 128, 256, 512],
                 stage_middle_channels=[64, 128, 256, 512],
                 blocks=[6, 12, 24, 16],
                 type='33', # 33 basic or 131 bottleneck
                 downsample_ration=[0.5, 0.5, 0.5, 0.5],
                 downsample_first=False,
                 norm='bn',
                 active='relu',
                 gn_c=8,
                 first_conv=True,
                 first_conv_channel=64,
                 dim=2):
        super().__init__()

        self.dim = dim
        self.first_conv = first_conv
        self.blocks = blocks
        self.downsample_ration =downsample_ration
        self.downsample_first =downsample_first

        # first_conv layer
        if first_conv:
            self.first_conv = ConvNormActive(in_channels, first_conv_channel, stride=2, kernel_size=7, norm=norm, active=active, gn_c=gn_c, dim=dim, padding=1)
        else:
            self.first_conv = None

        # res stages
        self.Res_stage = nn.ModuleList([])
        for stage_index in range(len(stage_output_channels)):
            if stage_index == 0:
                in_channel = first_conv_channel if first_conv else in_channels
                middle_channel = stage_middle_channels[stage_index]
                out_channel = stage_output_channels[stage_index]
                self.Res_stage.append(ResStage(type, in_channel, middle_channel, out_channel, block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))
            else:
                in_channel = stage_output_channels[stage_index-1]
                middle_channel = stage_middle_channels[stage_index]
                out_channel = stage_output_channels[stage_index]
                self.Res_stage.append(ResStage(type, in_channel, middle_channel, out_channel, block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))

    def forward(self, x):
        """
        # demo 1D
        dim = 1
        in_channels = 3
        input = torch.ones([2,in_channels, 128])
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]


        # demo 2D
        dim = 2
        in_channels = 3
        input = torch.ones([2,in_channels, 128, 128])
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256, 512],blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]


        # demo 3D
        dim = 3
        in_channels = 3
        input = torch.ones([2,in_channels, 64, 64, 64])
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128, 256],blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = ResNetEncoder(in_channels, stage_output_channels=[64, 128],blocks=[6, 12], downsample_ration=[0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]

        """
        if self.first_conv is not None:
            f = self.first_conv(x)
        else:
            f = x

        stage_features = []
        for stage_index, downsample_ratio in enumerate(self.downsample_ration):
            if self.downsample_first:
                f = resizeTensor(f, scale_factor=downsample_ratio)
                f = self.Res_stage[stage_index](f)
            else:
                f = self.Res_stage[stage_index](f)
                f = resizeTensor(f, scale_factor=downsample_ratio)
            stage_features.append(f)

        return stage_features


def ResNet18Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet18Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet18Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 3D
    dim = 3
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128, 128])
    encoder = ResNet18Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         blocks=[2, 2, 2, 2],
                         type='33',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


def ResNet34Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet34Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet34Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         blocks=[3, 4, 6, 3],
                         type='33',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


def ResNet50Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet50Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet50Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         stage_output_channels=[256, 512, 1024, 2048],
                         stage_middle_channels=[64, 128, 256, 512],
                         blocks=[3, 4, 6, 3],
                         type='131',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


def ResNet101Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet101Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet101Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         stage_output_channels=[256, 512, 1024, 2048],
                         stage_middle_channels=[64, 128, 256, 512],
                         blocks=[3, 4, 23, 3],
                         type='131',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


def ResNet152Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet152Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet152Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         stage_output_channels=[256, 512, 1024, 2048],
                         stage_middle_channels=[64, 128, 256, 512],
                         blocks=[3, 8, 36, 3],
                         type='131',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


def ResNet200Encoder(in_channels, dim):
    """
    # demo 1D
    dim = 1
    in_channels = 3
    input = torch.ones([2,in_channels, 128])
    encoder = ResNet200Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 3
    input = torch.ones([2,in_channels, 128, 128])
    encoder = ResNet200Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return ResNetEncoder(in_channels,
                         stage_output_channels=[256, 512, 1024, 2048],
                         stage_middle_channels=[64, 128, 256, 512],
                         blocks=[3, 24, 36, 3],
                         type='131',
                         downsample_ration=[0.5, 0.5, 0.5, 0.5],
                         dim=dim)


# densenet
class DenseNetEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 init_channels=64,
                 growth_rate=32,
                 bn_size=4,
                 blocks=[6, 12, 24, 16],
                 downsample_ration=[0.5, 0.5, 0.5, 0.5],
                 downsample_first=False,
                 norm='bn',
                 active='relu',
                 gn_c=8,
                 first_conv=True,
                 dim=2):
        super().__init__()

        self.dim =dim
        self.first_conv =first_conv
        self.blocks = blocks
        self.downsample_ration =downsample_ration
        self.downsample_first =downsample_first

        # first_conv layer
        if first_conv:
            self.first_conv = ConvNormActive(in_channels, init_channels, stride=2, kernel_size=7, norm=norm, active=active, gn_c=gn_c, dim=dim, padding=1)
        else:
            self.first_conv = None

        # denseblocks
        self.dense_blocks = nn.ModuleList([])
        for block_index, blocks_num in enumerate(blocks):
            if block_index == 0:
                num_features = init_channels if first_conv else in_channels
                self.dense_blocks.append(DenseBlock(blocks_num, num_features, growth_rate, bn_size, norm=norm, active=active, gn_c=gn_c, dim=dim))
                num_features = num_features + blocks_num * growth_rate
            else:
                num_features = num_features // 2
                self.dense_blocks.append(DenseBlock(blocks_num, num_features, growth_rate, bn_size, norm=norm, active=active, gn_c=gn_c, dim=dim))
                num_features = num_features + blocks_num * growth_rate

        # transation layers
        self.transition_layers = nn.ModuleList([])
        for block_index, blocks_num in enumerate(blocks):
            if block_index == 0:  # first
                num_features = init_channels if first_conv else in_channels
                num_features = num_features + blocks_num * growth_rate
                self.transition_layers.append(NormActiveConv(num_features, num_features // 2, norm=norm, active=active, gn_c=gn_c, dim=dim))
            elif block_index == len(blocks)-1:  # last
                self.transition_layers.append(nn.Identity())
            else:
                num_features = num_features // 2
                num_features = num_features + blocks_num * growth_rate
                self.transition_layers.append(NormActiveConv(num_features, num_features // 2, norm=norm, active=active, gn_c=gn_c, dim=dim))

    def forward(self, x):
        """
        # demo 1D
        dim = 1
        in_channels = 3
        input = torch.ones([2,in_channels, 128])
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]


        # demo 2D
        dim = 2
        in_channels = 3
        input = torch.ones([2,in_channels, 128, 128])
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.5, 0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], downsample_ration=[0.8, 0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.8, 0.8, 0.8], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]


        # demo 3D
        dim = 3
        in_channels = 3
        input = torch.ones([2,in_channels, 64, 64, 64])
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=False, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=True, first_conv=True, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.5, 0.5, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12, 24], downsample_ration=[0.3, 0.3, 0.5], downsample_first=False, first_conv=False, dim=dim)
        encoder = DenseNetEncoder(in_channels, blocks=[6, 12], downsample_ration=[0.3, 0.3], downsample_first=False, first_conv=False, dim=dim)
        multi_scale_f = encoder(input)
        _ = [print(i.shape) for i in multi_scale_f]

        """
        if self.first_conv is not None:
            f = self.first_conv(x)
        else:
            f = x

        blocks_features = []
        for block_index, downsample_ratio in enumerate(self.downsample_ration):
            if self.downsample_first:
                f = resizeTensor(f, scale_factor=downsample_ratio)
                f = self.dense_blocks[block_index](f)
                f = self.transition_layers[block_index](f)
            else:
                f = self.dense_blocks[block_index](f)
                f = self.transition_layers[block_index](f)
                f = resizeTensor(f, scale_factor=downsample_ratio)
            blocks_features.append(f)

        return blocks_features


def DenseNet121Encoder(in_channels, dim=2):
    """
    # demo 1D
    dim = 1
    in_channels = 4
    input = torch.ones([2,in_channels, 128])
    encoder = DenseNet121Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 4
    input = torch.ones([2,in_channels, 128, 128])
    encoder = DenseNet121Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 3D
    dim = 3
    in_channels = 4
    input = torch.ones([2,in_channels, 128, 128, 128])
    encoder = DenseNet121Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return DenseNetEncoder(in_channels, blocks=[6, 12, 24, 16], dim=dim)


def DenseNet169Encoder(in_channels, dim=2):
    """
    # demo 1D
    dim = 1
    in_channels = 4
    input = torch.ones([2,in_channels, 128])
    encoder = DenseNet169Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 4
    input = torch.ones([2,in_channels, 128, 128])
    encoder = DenseNet169Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return DenseNetEncoder(in_channels, blocks=[6, 12, 32, 32], dim=dim)


def DenseNet201Encoder(in_channels, dim=2):
    """
    # demo 1D
    dim = 1
    in_channels = 4
    input = torch.ones([2,in_channels, 128])
    encoder = DenseNet201Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 4
    input = torch.ones([2,in_channels, 128, 128])
    encoder = DenseNet201Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return DenseNetEncoder(in_channels, blocks=[6, 12, 48, 32], dim=dim)


def DenseNet264Encoder(in_channels, dim=2):
    """
    # demo 1D
    dim = 1
    in_channels = 4
    input = torch.ones([2,in_channels, 128])
    encoder = DenseNet264Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    # demo 2D
    dim = 2
    in_channels = 4
    input = torch.ones([2,in_channels, 128, 128])
    encoder = DenseNet264Encoder(in_channels, dim=dim)
    multi_scale_f = encoder(input)
    _ = [print(i.shape) for i in multi_scale_f]

    """
    return DenseNetEncoder(in_channels, blocks=[6, 12, 64, 48], dim=dim)
