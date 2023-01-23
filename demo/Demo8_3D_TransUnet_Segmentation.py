import torch
import torch.nn as nn
from wama_modules.Encoder import ResNetEncoder
from wama_modules.Decoder import UNet_decoder
from wama_modules.Head import SegmentationHead
from wama_modules.utils import resizeTensor
from transformers import ViTModel
from wama_modules.utils import load_weights, tmp_class


class TransUnet(nn.Module):
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

        # neck
        neck_out_channel = 768
        transformer = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        configuration = transformer.config
        self.trans_size_3D = [8, 8, 4]
        self.trans_size = configuration.image_size = [
            self.trans_size_3D[0], self.trans_size_3D[1]*self.trans_size_3D[2]
        ]
        configuration.patch_size = [1, 1]
        configuration.num_channels = Encoder_f_channel_list[-1]
        configuration.encoder_stride = 1  # just for MAE decoder, otherwise this paramater is not used
        self.neck = ViTModel(configuration, add_pooling_layer=False)

        pretrained_weights = transformer.state_dict()
        pretrained_weights['embeddings.position_embeddings'] = self.neck.state_dict()[
            'embeddings.position_embeddings']
        pretrained_weights['embeddings.patch_embeddings.projection.weight'] = self.neck.state_dict()[
            'embeddings.patch_embeddings.projection.weight']
        pretrained_weights['embeddings.patch_embeddings.projection.bias'] = self.neck.state_dict()[
            'embeddings.patch_embeddings.projection.bias']
        self.neck = load_weights(self.neck, pretrained_weights)  # reload pretrained weights

        # decoder
        Decoder_f_channel_list = [32, 64, 128]
        self.decoder = UNet_decoder(
            in_channels_list=Encoder_f_channel_list[:-1]+[neck_out_channel],
            skip_connection=[True, True, True],
            out_channels_list=Decoder_f_channel_list,
            dim=dim)

        # seg head
        self.seg_head = SegmentationHead(
            label_category_dict,
            Decoder_f_channel_list[0],
            dim=dim)

    def forward(self, x):
        # encoder forward
        multi_scale_encoder = self.encoder(x)

        # neck forward
        neck_input = resizeTensor(multi_scale_encoder[-1], size=self.trans_size_3D)
        neck_input = neck_input.reshape(neck_input.shape[0], neck_input.shape[1], *self.trans_size)  # 3D to 2D
        f_neck = self.neck(neck_input)
        f_neck = f_neck.last_hidden_state
        f_neck = f_neck[:, 1:]  # remove class token
        f_neck = f_neck.permute(0, 2, 1)
        f_neck = f_neck.reshape(
            f_neck.shape[0],
            f_neck.shape[1],
            self.trans_size[0],
            self.trans_size[1]
        )  # reshape
        f_neck = f_neck.reshape(f_neck.shape[0], f_neck.shape[1], *self.trans_size_3D)  # 2D to 3D
        f_neck = resizeTensor(f_neck, size=multi_scale_encoder[-1].shape[2:])
        multi_scale_encoder[-1] = f_neck

        # decoder forward
        multi_scale_decoder = self.decoder(multi_scale_encoder)
        f_for_seg = resizeTensor(multi_scale_decoder[0], size=x.shape[2:])

        # seg_head forward
        logits = self.seg_head(f_for_seg)
        return logits


if __name__ == '__main__':
    x = torch.ones([2, 1, 128, 128, 96])
    label_category_dict = dict(organ=3, tumor=4)
    model = TransUnet(in_channel=1, label_category_dict=label_category_dict, dim=3)
    with torch.no_grad():
        logits = model(x)
    print('multi_label predicted logits')
    _ = [print('logits of ', key, ':', logits[key].shape) for key in logits.keys()]

    # out
    # multi_label predicted logits
    # logits of  organ : torch.Size([2, 3, 128, 128, 96])
    # logits of  tumor : torch.Size([2, 4, 128, 128, 96])

