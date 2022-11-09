'''SqueezeNet in PyTorch.

See the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" for more details.
'''

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
        	out += x
        out = self.relu(out)

        return out


class SqueezeNet(nn.Module):

    def __init__(self,):
        super(SqueezeNet, self).__init__()
        # if version not in [1.0, 1.1]:
        #     raise ValueError("Unsupported SqueezeNet version {version}:"
        #                      "1.0 or 1.1 expected".format(version=version))

        # if version == 1.1:
        if True:
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(1,2,2), padding=(1,1,1)), # 0
                nn.BatchNorm3d(64), # 1
                nn.ReLU(inplace=True), # 2
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # todo 3
                Fire(64, 16, 64, 64), # 4
                Fire(128, 16, 64, 64, use_bypass=True), # 5
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # todo 6
                Fire(128, 32, 128, 128), # 7
                Fire(256, 32, 128, 128, use_bypass=True), # 8
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # todo 9
                Fire(256, 48, 192, 192), # 10
                Fire(384, 48, 192, 192, use_bypass=True), # 11
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # todo 12
                Fire(384, 64, 256, 256), # 13
                Fire(512, 64, 256, 256, use_bypass=True), # todo 14
            )
        # Final convolution is initialized differently form the rest

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        f_list = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in [3,6,9,12,14]:
                f_list.append(x)
        return f_list


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

    
def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SqueezeNet(**kwargs)
    return model


if __name__ == '__main__':
    model = SqueezeNet(version=1.1, sample_size = 112, sample_duration = 16, num_classes=600)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
