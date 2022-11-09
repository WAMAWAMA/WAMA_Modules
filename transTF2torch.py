import tensorflow as tf
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, densenet121, inception_v3
from keras.layers.convolutional import Conv2D
import keras
import collections

# outpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-ResNet50_notop_torch.pth"
# inputpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-ResNet50_notop.h5"
# torchnet = resnet50

outpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\pytorch\RadImageNet-DenseNet121_notop_torch.pth"
inputpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\tensorflow\RadImageNet-DenseNet121_notop.h5"
torchnet = densenet121

# outpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-InceptionV3_notop_torch.pth"
# inputpath = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-InceptionV3_notop.h5"
# torchnet = inception_v3


# def simple_test(net):
#     img = Image.open(testimg).convert('RGB')
#
#     trans_data = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#     ])
#
#     img = trans_data(img).unsqueeze(0)
#     out = net(img)
#     return out.squeeze(0)[0]


def keras_to_pyt(km, pm=None):
    weight_dict = dict()
    for layer in km.layers:
        if (type(layer) is Conv2D) and ('0' not in layer.get_config()['name']):
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            # weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1] as mean
        elif type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]

    if pm:
        pyt_state_dict = pm.state_dict()
        for key in pyt_state_dict.keys():
            pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
        pm.load_state_dict(pyt_state_dict)
        return pm
    return weight_dict


net = torchnet(num_classes=1)
# out = simple_test(net)
# print('before output is', out)

tf_keras_model = tf.keras.models.load_model(inputpath)
weights = tf_keras_model.get_weights()

weights = keras_to_pyt(tf_keras_model)
values = list(weights.values())
i = 0
for name, param in net.named_parameters():
    if 'conv' in name:
        print(name, param.data.shape, values[i].shape)
        # param.data = torch.tensor(values[i])
        # print('load weights:', values[i].shape)
        # print(param.data.shape, values[i].shape)
        # print(param.data.shape)
        # param.data = torch.tensor(values[i])
        i += 1
        if i == len(values):
            break

# out = simple_test(net)
# print('after output is', out)

torch.save(net.state_dict(), outpath)
