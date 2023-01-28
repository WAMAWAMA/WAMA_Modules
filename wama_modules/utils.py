import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pickle


class tmp_class():
    """
    for debug
    """
    def __init__(self,):
        super().__init__()


def resizeTensor(x, scale_factor=None, size=None):
    """
    resize for 1D\2D\3D tensor (1D → signal  2D → image, 3D → volume)

    :param x: 1D [bz,c,l] 2D [bz,c,w,h] 3D [bz,c,w,h,l]
    :param scale_factor: 1D [2.,] 2D [2.,2.,] 3D [3.,3.,3.,]
    :param size: 2D [256.,256,m]  or torch.ones([256,256]).shape
    :return:

    # 1D demo:
    x = torch.ones([3,1,256])
    y = torch.ones([3,1,128])
    x1 = resizeTensor(x, scale_factor=[2.,])
    print(x1.shape)
    x1 = resizeTensor(x, size=y.shape[-1:])
    print(x1.shape)

    # 2D demo:
    x = torch.ones([3,1,256,256])
    y = torch.ones([3,1,256,128])
    x1 = resizeTensor(x, scale_factor=[2.,2.])
    print(x1.shape)
    x1 = resizeTensor(x, size=y.shape[-2:])
    print(x1.shape)

    # 3D demo:
    x = torch.ones([3,1,256,256,256])
    y = torch.ones([3,1,256,128,128])
    x1 = resizeTensor(x, scale_factor=[2.,2.,2.])
    print(x1.shape)
    x1 = resizeTensor(x, size=y.shape[-3:])
    print(x1.shape)

    """
    if len(x.shape) == 3:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='linear',
                             align_corners=True)
    if len(x.shape) == 4:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='bicubic',
                             align_corners=True)
    elif len(x.shape) == 5:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='trilinear',
                             align_corners=True)


def tensor2array(tensor):
    return tensor.data.cpu().numpy()


def load_weights(model, state_dict, drop_modelDOT=False, silence=False):
    if drop_modelDOT:
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k[7:]] = v
        state_dict = new_dict
    net_dict = model.state_dict()  # model dict
    pretrain_dict = {k: v for k, v in state_dict.items()}  # pretrain dict
    InPretrain_InModel_dict = {k: v for k, v in state_dict.items() if k in net_dict.keys()}
    InPretrain_NotInModel_dict = {k: v for k, v in state_dict.items() if k not in net_dict.keys()}
    NotInPretrain_InModel_dict = {k: v for k, v in net_dict.items() if k not in state_dict.keys()}
    if not silence:
        print('-' * 200)
        print('keys ( Current model,C ) ', len(net_dict.keys()), net_dict.keys())
        print('keys ( Pre-trained  ,P ) ', len(pretrain_dict.keys()), pretrain_dict.keys())
        print('keys (   In C &   In P ) ', len(InPretrain_InModel_dict.keys()), InPretrain_InModel_dict.keys())
        print('keys ( NoIn C &   In P ) ', len(InPretrain_NotInModel_dict.keys()), InPretrain_NotInModel_dict.keys())
        print('keys (   In C & NoIn P ) ', len(NotInPretrain_InModel_dict.keys()), NotInPretrain_InModel_dict.keys())
        print('-' * 200)
        print('Pretrained keys :', len(InPretrain_InModel_dict.keys()), InPretrain_InModel_dict.keys())
        print('Non-Pretrained keys:', len(NotInPretrain_InModel_dict.keys()), NotInPretrain_InModel_dict.keys())
        print('-' * 200)
    net_dict.update(InPretrain_InModel_dict)
    model.load_state_dict(net_dict)
    return model


def MaxMinNorm(array, FirstDimBATCH = True):
    """
    :param array:
    :param FirstDimBATCH: bool, is the first dim batch?  True or False
    :return:

    # demo for numpy ndarray




    # demo for torch tensor





    """
    pass


def mat2gray(image):
    """
    归一化函数（线性归一化）
    :param image: ndarray
    :return:
    """
    # as dtype = np.float32
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-14)
    return image


def save_as_pkl(save_path, obj):
    data_output = open(save_path, 'wb')
    pickle.dump(obj, data_output)
    data_output.close()

def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data


import matplotlib.pyplot as plt
def show2D(img):
    plt.imshow(img)
    plt.show()

# try:
#     from mayavi import mlab
#     def show3D(img3D):
#         vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
#         mlab.colorbar(orientation='vertical')
#         mlab.show()
# except:
#     pass
