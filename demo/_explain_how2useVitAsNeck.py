import torch
from transformers import ViTConfig, ViTModel
from wama_modules.utils import load_weights, tensor2array


m = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
f = m(torch.ones([1, 3, 224, 224]), output_hidden_states=True)
f_last = f.last_hidden_state
print(f_last.shape)
f_cls_token = (torch.squeeze(f_last[:,0])).data.cpu().numpy()
f_cls_token = list(f_cls_token)


configuration = m.config
configuration.image_size = [16, 8]
configuration.patch_size = [1, 1]
configuration.num_channels = 1
configuration.encoder_stride = 1  # just for MAE decoder, otherwise this paramater is not used
m1 = ViTModel(configuration, add_pooling_layer=False)

f = m1(torch.ones([2, 1, 16, 8]), output_hidden_states=True)


f_list = f.hidden_states  # For transformer, should use reshaped_hidden_states
_ = [print(i.shape) for i in f_list]

f_last = f.last_hidden_state
f_last = f_last[:, 1:]
f_last = f_last.permute(0, 2, 1)
f_last = f_last.reshape(f_last.shape[0], f_last.shape[1], configuration.image_size[0], configuration.image_size[1])
print('spatial f_last:', f_last.shape)


# reload weights

m = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
weights = m.state_dict()
weights['embeddings.position_embeddings'] = m1.state_dict()['embeddings.position_embeddings']
weights['embeddings.patch_embeddings.projection.weight'] = m1.state_dict()['embeddings.patch_embeddings.projection.weight']
weights['embeddings.patch_embeddings.projection.bias'] = m1.state_dict()['embeddings.patch_embeddings.projection.bias']


m1 = load_weights(m1, weights)


# test: spatial visualization
m1 = ViTModel(configuration, add_pooling_layer=False)

input = torch.ones([2, 1, 16, 8])*100
input[:,:,8:] = input[:,:,8:]*0.
input[:,:,:3] = input[:,:,:3]*0.
input[:,:,:,:3] = input[:,:,:,:3]*0.
f = m1(input, output_hidden_states=True)
f_last = f.last_hidden_state
f_last = f_last[:, 1:]
f_last = f_last.permute(0, 2, 1)
f_last = f_last.reshape(f_last.shape[0], f_last.shape[1], configuration.image_size[0], configuration.image_size[1])
print('spatial f_last:', f_last.shape)
print(f_last.max())
print(f_last.min())

def tensor2numpy(tensor):
    return tensor.data.cpu().numpy()
import numpy as np
def mat2gray(image):
    """
    归一化函数（线性归一化）
    :param image: ndarray
    :return:
    """
    # as dtype = np.float32
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image)-np.min(image)+ 1e-14)
    return image

import matplotlib.pyplot as plt
def show2D(img):
    plt.imshow(img)
    plt.show()


# the two image should be aligned in space
show2D(tensor2numpy(f_last[0,0]))
show2D(tensor2numpy(input[0,0]))

