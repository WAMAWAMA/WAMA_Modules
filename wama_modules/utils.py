import torch.nn as nn
import torch
import torch.nn.functional as F



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

