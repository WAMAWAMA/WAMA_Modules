import torch
import torch.nn as nn
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Head import ClassificationHead
from wama_modules.BaseModule import GlobalMaxPool


class Model(nn.Module):
    def __init__(self, in_channel, label_category_dict, dim=2):
        super().__init__()
        # encoder
        f_channel_list = [64, 128, 256, 512]
        self.encoder = ResNetEncoder(
            in_channel,
            stage_output_channels=f_channel_list,
            stage_middle_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            type='131',
            downsample_ration=[0.5, 0.5, 0.5, 0.5],
            dim=dim)
        # cls head
        self.cls_head = ClassificationHead(label_category_dict, f_channel_list[-1])
        self.pooling = GlobalMaxPool()

    def forward(self, x):
        f = self.encoder(x)
        logits = self.cls_head(self.pooling(f[-1]))
        return logits


if __name__ == '__main__':
    x = torch.ones([2, 1, 64, 64, 64])
    label_category_dict = dict(is_malignant=4)
    model = Model(in_channel=1, label_category_dict=label_category_dict, dim=3)
    logits = model(x)
    print('single-label predicted logits')
    _ = [print('logits of ', key, ':', logits[key].shape) for key in logits.keys()]

    # output 👇
    # single-label predicted logits
    # logits of  is_malignant : torch.Size([2, 4])
