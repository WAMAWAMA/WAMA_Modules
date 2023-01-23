import torch
import torch.nn as nn
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Decoder import UNet_decoder
from wama_modules.Head import SegmentationHead
from wama_modules.utils import resizeTensor


class Model(nn.Module):
    def __init__(self, in_channel, label_category_dict, dim=2):
        super().__init__()
        # encoder
        Encoder_f_channel_list = [64, 128, 256, 512]
        self.encoder = ResNetEncoder(
            in_channel,
            stage_output_channels=Encoder_f_channel_list,
            stage_middle_channels=Encoder_f_channel_list,
            blocks=[1, 2, 3, 4],
            type='131',
            downsample_ration=[0.5, 0.5, 0.5, 0.5],
            dim=dim)
        # decoder
        Decoder_f_channel_list = [32, 64, 128]
        self.decoder = UNet_decoder(
            in_channels_list=Encoder_f_channel_list,
            skip_connection=[False, True, True],
            out_channels_list=Decoder_f_channel_list,
            dim=dim)
        # seg head
        self.seg_head = SegmentationHead(
            label_category_dict,
            Decoder_f_channel_list[0],
            dim=dim)

    def forward(self, x):
        multi_scale_f1 = self.encoder(x)
        multi_scale_f2 = self.decoder(multi_scale_f1)
        f_for_seg = resizeTensor(multi_scale_f2[0], size=x.shape[2:])
        logits = self.seg_head(f_for_seg)
        return logits


if __name__ == '__main__':
    x = torch.ones([2, 1, 128, 128, 128])
    label_category_dict = dict(organ=3)
    model = Model(in_channel=1, label_category_dict=label_category_dict, dim=3)
    logits = model(x)
    print('multi_label predicted logits')
    _ = [print('logits of ', key, ':', logits[key].shape) for key in logits.keys()]

    # out
    # multi_label predicted logits
    # logits of  organ : torch.Size([2, 3, 128, 128, 128])
