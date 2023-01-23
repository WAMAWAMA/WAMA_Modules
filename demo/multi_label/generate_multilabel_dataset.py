"""
generate example multi_label dataset
"""
import numpy as np


label_category_dict = dict(
    water=2,  # binary class label
    milk=2,  # binary class label
    cow=2,  # binary class label
    big_white_fish=2,  # binary class label
    grass=2,  # binary class label
    weather=5,  # 5-class label
)
label_name = list(label_category_dict.keys())
print('label num :',len(label_name))
_ = [print('-'*4, key, ': class num = ', label_category_dict[key]) for key in label_category_dict.keys()]

img_channel = 2
dataset = [
    dict(  # case1
        img_1D=np.ones([128, img_channel]),  # which is also a 1D signal
        img_2D=np.ones([128, 128, img_channel]),
        img_3D=np.ones([64, 64, 64, img_channel]),
        label_value=[1, 1, 1, 0, 0, 1],  # 0=negative, 1=positive
        label_known=[1, 1, 1, 1, 1, 1],  # 0=unknown/missing, 1=known
    ),
    dict(  # case2
        img_1D=np.ones([128, img_channel]),
        img_2D=np.ones([128, 128, img_channel]),
        img_3D=np.ones([64, 64, 64, img_channel]),
        label_value=[1, 0, 0, 0, 1, 1],
        label_known=[1, 1, 1, 0, 0, 0],
    ),
    dict(  # case3
        img_1D=np.ones([128, img_channel]),
        img_2D=np.ones([128, 128, img_channel]),
        img_3D=np.ones([64, 64, 64, img_channel]),
        label_value=[1, 1, 0, 1, 1, 0],
        label_known=[1, 1, 0, 1, 0, 0],
    ),
]
