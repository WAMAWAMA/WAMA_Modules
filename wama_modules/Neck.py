import torch.nn as nn
import torch
from wama_modules.BaseModule import ConvNormActive
from wama_modules.utils import resizeTensor


class FPN(nn.Module):
    def __init__(self,
                 in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddSmall2Big',  # AddSmall2Big or AddBig2Small(much better or classification tasks)
                 dim=2,
                 ):
        super().__init__()
        self.mode = mode

        self.conv1_list = nn.ModuleList([
            ConvNormActive(in_channels, c1, kernel_size=1, norm=norm, active=active, gn_c=gn_c, dim=dim, padding=0)
            for in_channels in in_channels_list
        ])
        self.conv2_list = nn.ModuleList([
            ConvNormActive(c1, c2, kernel_size=3, norm=norm, active=active, gn_c=gn_c, dim=dim, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, x_list):
        """
        :param x_list: multi scale feature maps, from shallow(big size) to deep(small size)
        :return:

        # demo
        # 1D
        x_list = [
            torch.ones([3,16,32]),
            torch.ones([3,32,24]),
            torch.ones([3,64,16]),
            torch.ones([3,128,8]),
        ]
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddSmall2Big',
                 dim=1,)
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddBig2Small', # revserse, for classification
                 dim=1,)
        f_list = fpn(x_list)
        _ = [print(i.shape) for i in x_list]
        _ = [print(i.shape) for i in f_list]

        # 2D
        x_list = [
            torch.ones([3,16,32,32]),
            torch.ones([3,32,24,24]),
            torch.ones([3,64,16,16]),
            torch.ones([3,128,8,8]),
        ]
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddSmall2Big',
                 dim=2,)
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddBig2Small', # revserse, for classification
                 dim=2,)
        f_list = fpn(x_list)
        _ = [print(i.shape) for i in x_list]
        _ = [print(i.shape) for i in f_list]


        # 3D
        x_list = [
            torch.ones([3,16,32,32,32]),
            torch.ones([3,32,24,24,24]),
            torch.ones([3,64,16,16,16]),
            torch.ones([3,128,8,8,8]),
        ]
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddSmall2Big',
                 dim=3,)
        fpn = FPN(in_channels_list=[16, 32, 64, 128],
                 c1=128,
                 c2=256,
                 active='relu',
                 norm='bn',
                 gn_c=8,
                 mode='AddBig2Small', # revserse, for classification
                 dim=3,)
        f_list = fpn(x_list)
        _ = [print(i.shape) for i in x_list]
        _ = [print(i.shape) for i in f_list]

        """
        f_list = [self.conv1_list[index](f) for index, f in enumerate(x_list)]

        if self.mode == 'AddSmall2Big':
            f_list_2 = []
            x = f_list[-1]
            f_list_2.append(x)
            for index in range(len(f_list)-1):
                # print(f_list[-(index+2)].shape[2:])
                x = f_list[-(index + 2)] + resizeTensor(x, size=f_list[-(index+2)].shape[2:])
                f_list_2.append(x)
            f_list_2 = f_list_2[::-1]
        elif self.mode == 'AddBig2Small':
            f_list_2 = []
            x = f_list[0]
            f_list_2.append(x)
            for index in range(len(f_list) - 1):
                # print(f_list[index + 1].shape[2:])
                x = f_list[index + 1] + resizeTensor(x, size=f_list[index + 1].shape[2:])
                f_list_2.append(x)

        return_list = [self.conv2_list[index](f) for index, f in enumerate(f_list_2)]

        return return_list




#  UCtrans的





