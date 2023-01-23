import torch
import torch.nn as nn
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Decoder import UNet_decoder
from wama_modules.Head import SegmentationHead, ClassificationHead
from wama_modules.utils import resizeTensor
from wama_modules.BaseModule import GlobalMaxPool


class Model(nn.Module):
    def __init__(self,
                 in_channel,
                 seg_label_category_dict,
                 cls_label_category_dict,
                 dim=2):
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
            seg_label_category_dict,
            Decoder_f_channel_list[0],
            dim=dim)
        # cls head
        self.cls_head = ClassificationHead(cls_label_category_dict, Encoder_f_channel_list[-1])

        # pooling
        self.pooling = GlobalMaxPool()

    def forward(self, x):
        # get encoder features
        multi_scale_encoder = self.encoder(x)
        # get decoder features
        multi_scale_decoder = self.decoder(multi_scale_encoder)
        # perform segmentation
        f_for_seg = resizeTensor(multi_scale_decoder[0], size=x.shape[2:])
        seg_logits = self.seg_head(f_for_seg)
        # perform classification
        cls_logits = self.cls_head(self.pooling(multi_scale_encoder[-1]))
        return seg_logits, cls_logits

if __name__ == '__main__':
    x = torch.ones([2, 1, 128, 128, 128])
    seg_label_category_dict = dict(organ=3, tumor=2)
    cls_label_category_dict = dict(shape=4, color=3, other=13)
    model = Model(
        in_channel=1,
        cls_label_category_dict=cls_label_category_dict,
        seg_label_category_dict=seg_label_category_dict,
        dim=3)
    seg_logits, cls_logits = model(x)
    print('multi_label predicted logits')
    _ = [print('seg logits of ', key, ':', seg_logits[key].shape) for key in seg_logits.keys()]
    print('-'*30)
    _ = [print('cls logits of ', key, ':', cls_logits[key].shape) for key in cls_logits.keys()]

    # out
    # multi_label predicted logits
    # seg logits of  organ : torch.Size([2, 3, 128, 128, 128])
    # seg logits of  tumor : torch.Size([2, 2, 128, 128, 128])
    # ------------------------------
    # cls logits of  shape : torch.Size([2, 4])
    # cls logits of  color : torch.Size([2, 3])
    # cls logits of  other : torch.Size([2, 13])


