import torchvision
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from wama_modules.BaseModule import GlobalAvgPool
BasicBlock = GlobalAvgPool()
print(1)
print(1)
print(1)
print(1)
print(1)
print(1)

import torch
from wama_modules.thirdparty_lib.ResNets3D_kenshohara.models.resnet import generate_model
m = generate_model(18, n_classes = 1039)
m.load_state_dict(torch.load(r"D:\pretrainedweights\kenshohara_ResNets3D\weights\r3d18_KM_200ep.pth", map_location='cpu')['state_dict'])




import torch
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ssl')
m = get_encoder('name', in_channels=3, depth=5, weights='ssl')
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ss')
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]

import timm
m = timm.create_model(
    'adv_inception_v3',
    features_only=True,
    pretrained=False,)
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]
