import torch.nn as nn
import torch
import torch.nn.functional as F
from wama_modules.utils import tensor2array, resizeTensor, tmp_class
from wama_modules.BaseModule import ConvNormActive


class UNet_decoder(nn.Module):
    def __init__(self,
                 in_channels_list=[64, 128, 256, 512],  # from shallow to deep
                 skip_connection=[True, True, True],  # from shallow to deep
                 out_channels_list=[12, 13, 14],  # from shallow to deep
                 norm='bn',
                 gn_c=8,
                 dim=2,
                 ):
        super().__init__()
        self._skip_connection = skip_connection[::-1]  # from deep to shallow
        _skip_channels_list = in_channels_list[-2::-1]  # from deep to shallow [256, 128, 64]
        _out_channels_list = out_channels_list[::-1]  # from deep to shallow [14, 13, 12]
        _in_conv_list = [in_channels_list[-1]] + _out_channels_list[:-1]  # from deep to shallow
        self.docoder_conv_list = nn.ModuleList([])
        for stage, _out_channels in enumerate(_out_channels_list):
            if self._skip_connection[stage]:
                _in_channel = _in_conv_list[stage] + _skip_channels_list[stage]
            else:
                _in_channel = _in_conv_list[stage]
            _out_channel = _out_channels_list[stage]
            self.docoder_conv_list.append(
                nn.Sequential(
                    ConvNormActive(_in_channel, _out_channel, kernel_size=3, norm=norm, gn_c=gn_c, dim=dim),
                    ConvNormActive(_out_channel, _out_channel, kernel_size=3, norm=norm, gn_c=gn_c, dim=dim),
                )
            )

    def forward(self, f_list):
        """
        :return: decoder_f_list, feature list from shallow to deep, and decoder_f_list[0] can be used for seg head

        # demo

        # 1D -------------------------------------------------------------
        f_list = [
            torch.ones([3,64,128]),
            torch.ones([3,128,64]),
            torch.ones([3,256,32]),
            torch.ones([3,512,8]),
        ]

        decoder = UNet_decoder(
            in_channels_list=[64, 128, 256, 512],  # from shallow to deep
            skip_connection=[False, True, True],  # from shallow to deep
            out_channels_list=[12, 13, 14],  # from shallow to deep
            norm='bn',
            gn_c=8,
            dim=1
        )

        decoder_f_list = decoder(f_list)
        _ = [print(i.shape) for i in decoder_f_list]

        # 2D -------------------------------------------------------------
        f_list = [
            torch.ones([3,64,128,128]),
            torch.ones([3,128,64,64]),
            torch.ones([3,256,32,32]),
            torch.ones([3,512,8,8]),
        ]

        decoder = UNet_decoder(
            in_channels_list=[64, 128, 256, 512],  # from shallow to deep
            skip_connection=[False, True, True],  # from shallow to deep
            out_channels_list=[12, 13, 14],  # from shallow to deep
            norm='bn',
            gn_c=8,
            dim=2
        )

        decoder_f_list = decoder(f_list)
        _ = [print(i.shape) for i in decoder_f_list]

        # 3D -------------------------------------------------------------
        f_list = [
            torch.ones([3,64,128,128,128]),
            torch.ones([3,128,64,64,64]),
            torch.ones([3,256,32,32,32]),
            torch.ones([3,512,8,8,8]),
        ]

        decoder = UNet_decoder(
            in_channels_list=[64, 128, 256, 512],  # from shallow to deep
            skip_connection=[False, True, True],  # from shallow to deep
            out_channels_list=[12, 13, 14],  # from shallow to deep
            norm='bn',
            gn_c=8,
            dim=3
        )

        decoder_f_list = decoder(f_list)
        _ = [print(i.shape) for i in decoder_f_list]

        """
        _f_list = f_list[::-1]
        feature = _f_list[0]
        _f_list = _f_list[1:]
        decoder_f_list = []
        for stage, conv in enumerate(self.docoder_conv_list):
            if self._skip_connection[stage]:
                _in_feature = torch.cat([resizeTensor(feature, size=_f_list[stage].shape[2:]), _f_list[stage]], 1)
            else:
                _in_feature = resizeTensor(feature, size=_f_list[stage].shape[2:])
            feature = conv(_in_feature)
            decoder_f_list.append(feature)
        decoder_f_list = decoder_f_list[::-1]
        return decoder_f_list  # from shallow to deep, and decoder_f_list[0] can be used for seg head

# psp


# deeplabv3+
# try this https://blog.csdn.net/m0_51436734/article/details/124073901


# NestedUNet(Unet++)









