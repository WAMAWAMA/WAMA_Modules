# ωαмα m⚙️dules

(🚧 still under building, but feel free to try current module implementations~)

*A PyTorch module library for building 1D/2D/3D networks flexibly ~*

Highlights (*Simple-to-use & Function-rich!*)
 - Simple code that show whole forward processes succinctly
 - Output rich features and attention map for fast reuse
 - Support 1D / 2D / 3D networks (CNNs, GNNs, Transformers...)
 - Easy and flexible to integrate with any other network
 - 🚀 Abundant Pretrained weights: Including 80000+ `2D weights` and 80+ `3D weights`

## 1. Installation
🔥 [wama_modules](https://github.com/WAMAWAMA/wama_modules) `Basic` `1D` `2D` `3D`

Install `wama_modules` with command ↓
```
pip install git+https://github.com/WAMAWAMA/wama_modules.git
```

 *Or you can directly copy the `wama_modules` folder to use

💧 [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) `Optional` `2D` `100+ pretrained weights`

<details>
<summary> Introduction and installation command </summary>

`segmentation_models_pytorch` (called `smp`)
is a 2D CNN lib includes many backbones and decoders, which is highly recommended to install for cooperating with this library. 

Our code have already contained `smp`, but you can still install the latest version with the code below.

Install with pip：
```
pip install segmentation-models-pytorch
```
Install the latest version:
```
pip install git+https://github.com/rwightman/pytorch-image-models.git
```

</details>

💧 [transformers](https://github.com/huggingface/transformers) `Optional` `2D` `80000+ pretrained weights`

<details>
<summary> Introduction and installation command </summary>

`transformer` is a lib includes abundant CNN and Transformer structures, which is highly recommended to install for cooperating with this library. 

Install `transformer` use ↓
```
pip install transformers
```


</details>

💧 [timm](https://github.com/rwightman/pytorch-image-models) `Optional` `2D` `400+ pretrained weights` 

<details>
<summary> Introduction and installation command </summary>

`timm` is a lib includes abundant CNN and Transformer structures, which is highly recommended to install for cooperating with this library. 
Install *transformer* use ↓

Install with pip:
```
pip install timm
```
Install the latest version:
```
pip install git+https://github.com/rwightman/pytorch-image-models.git
```
</details>


## 2. Update list
 - 2022/11/5:  Open the source code, version `v0.0.1-beta`
 - ...


## 3. Main modules and network architectures
Here I'll give an overview of this repo

## 4. Guideline 1: Build networks modularly
How to build a network modularly? 

The answer is a paradigm of building networks:

***'Design architecture according to tasks, pick modules according to architecture'***

So, network architectures for different tasks can be viewed modularly such as:
 - vgg = vgg_encoder + cls_head
 - Unet = encoder + decoder + seg_ead
 - resnet = resnet_encoder + cls_head
 - densenet = densenet_encoder + cls_head
 - a multi-task net for classification and segmentation = encoder + decoder + cls_head + seg_head



For example, build a 3D resnet50 


```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```





Here are more demos shown below ↓ (Click to view codes, or visit the `demo` folder)



<details>
<summary> Demo1: Build a 2D vgg16  </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo2: Build a 3D resnet50  </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>



<details>
<summary> Demo3: Build a 3D densenet121  </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo4: Build a Unet  </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo5: Build a Unet with a resnet50 encoder  </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo6: Build a Unet with a resnet50 encoder and a FPN </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo7: Build a multi-task model for segmentation and classification</summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>



<details>
<summary> Demo8: Build a C-tran model for multi-label classification</summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo9: Build a Q2L model for multi-label classification</summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo10: Build a ML-Decoder model for multi-label classification</summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo11: Build a ML-GCN model for multi-label classification</summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo12: Build a UCTransNet model for segmentation </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo13: Build a model for multiple inputs (1D signal and 2D image) </summary>

```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo14: Build a 2D Unet with pretrained Resnet50 encoder (1D signal and 2D image) </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>


<details>
<summary> Demo15: Build a 3D DETR model for object detection </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>

<details>
<summary> Demo16: Build a 3D VGG with SE-attention module for multi-instanse classification </summary>
 
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```
</details>



## 5. Guideline 2: Use pretrained weights

All pretrained weights are from third-party codes or repos

current pretrained support: (这里给一个表格，来自哪里，多少权重，预训练数据类型，2D还是3D）)
 - 2D: smp, timm, radimagenet...
 - 3D: medicalnet, 3D resnet, 3D densenet...


### 5.1  smp encoders `2D`

smp (119 pretrained weights)

```python
import torch
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ssl')
m = get_encoder('name', in_channels=3, depth=5, weights='ssl')
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ss')
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]
```

### 5.2  timm encoders `2D`
timm (400+ pretrained weights)
```python
import timm
m = timm.create_model(
    'adv_inception_v3',
    features_only=True,
    pretrained=False,)
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]
```
### 5.3  Transformers (🤗 Huggingface )  `2D`

transformers, supper powered by Huggingface ( with 80000+ pretrained weights)

```python
import torch
from transformers import ConvNextModel
from wama_modules.utils import load_weights
# Initializing a model (with random weights) from the convnext-tiny-224 style configuration
m = ConvNextModel.from_pretrained('facebook/convnext-base-224-22k')
f = m(torch.ones([2,3,224,224]), output_hidden_states=True)
f_list = f.hidden_states
_ = [print(i.shape) for i in f_list]

weights = m.state_dict()
m1 = ConvNextModel(m.config)
m = load_weights(m, weights)


import torch
from transformers import SwinModel
from wama_modules.utils import load_weights

m = SwinModel.from_pretrained('microsoft/swin-base-patch4-window12-384')
f = m(torch.ones([2,3,384,384]), output_hidden_states=True)
f_list = f.reshaped_hidden_states # For transformer, should use reshaped_hidden_states
_ = [print(i.shape) for i in f_list]

weights = m.state_dict()
m1 = SwinModel(m.config)
m = load_weights(m, weights)



```

### 5.2  radimagenet `2D` `medical image`
???


### 5.3 ResNets3D_kenshohara `3D` `video`
3D ResNets3D_kenshohara (21 weights)
```python
 import torch
    from wama_modules.thirdparty_lib.ResNets3D_kenshohara.resnet import generate_model
    from wama_modules.utils import load_weights
    m = generate_model(18)
    pretrain_path = r"D:\pretrainedweights\ResNets3D_kenshohara\weights\resnet\r3d18_KM_200ep.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights)
    f_list = m(torch.ones([2,3,64,64,64]))
    _ = [print(i.shape) for i in f_list]


    import torch
    from wama_modules.thirdparty_lib.ResNets3D_kenshohara.resnet2p1d import generate_model
    from wama_modules.utils import load_weights
    m = generate_model(18)
    pretrain_path = r"D:\pretrainedweights\ResNets3D_kenshohara\weights\resnet2p1d\r2p1d18_K_200ep.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights)
    f_list = m(torch.ones([2,3,64,64,64]))
    _ = [print(i.shape) for i in f_list]
```
### 5.3 VC3D_kenshohara `3D` `video`
3D VC3D_kenshohara (13 weights)
```python
 import torch
    from wama_modules.thirdparty_lib.VC3D_kenshohara.resnet import generate_model
    from wama_modules.utils import load_weights
    m = generate_model(18)
    pretrain_path = r"D:\pretrainedweights\VC3D_kenshohara\VC3D_weights\resnet\resnet-18-kinetics.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights, drop_modelDOT=True)
    f_list = m(torch.ones([2,3,64,64,64]))
    _ = [print(i.shape) for i in f_list]

    import torch
    from wama_modules.thirdparty_lib.VC3D_kenshohara.resnext import generate_model
    from wama_modules.utils import load_weights
    m = generate_model(101)
    pretrain_path = r"D:\pretrainedweights\VC3D_kenshohara\VC3D_weights\resnext\resnext-101-64f-kinetics.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights, drop_modelDOT=True)
    f_list = m(torch.ones([2,3,64,64,64]))
    _ = [print(i.shape) for i in f_list]

    import torch
    from wama_modules.thirdparty_lib.VC3D_kenshohara.wide_resnet import generate_model
    from wama_modules.utils import load_weights
    m = generate_model()
    pretrain_path = r"D:\pretrainedweights\VC3D_kenshohara\VC3D_weights\wideresnet\wideresnet-50-kinetics.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights, drop_modelDOT=True)
    f_list = m(torch.ones([2,3,64,64,64]))
    _ = [print(i.shape) for i in f_list]
```
### 5.3 ??? `3D` `video`
???

### 5.3 ??? `3D` `medical image`
???





## 6. All modules and functions

### 6.1 `wama_modules.BaseModule`

#### 6.1.1 Pooling
 - `GlobalAvgPool` Global average pooling
 - `GlobalMaxPool` Global maximum pooling
 - `GlobalMaxAvgPool` GlobalMaxAvgPool = (GlobalAvgPool + GlobalMaxPool) / 2.

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
# import libs
import torch
from wama_modules.BaseModule import GlobalAvgPool, GlobalMaxPool, GlobalMaxAvgPool

# make tensor
inputs1D = torch.ones([3,12,13]) # 1D
inputs2D = torch.ones([3,12,13,13]) # 2D
inputs3D = torch.ones([3,12,13,13,13]) # 3D

# build layer
GAP = GlobalAvgPool()
GMP = GlobalMaxPool()
GAMP = GlobalMaxAvgPool()

# test GAP & GMP & GAMP
print(inputs1D.shape, GAP(inputs1D).shape)
print(inputs2D.shape, GAP(inputs2D).shape)
print(inputs3D.shape, GAP(inputs3D).shape)

print(inputs1D.shape, GMP(inputs1D).shape)
print(inputs2D.shape, GMP(inputs2D).shape)
print(inputs3D.shape, GMP(inputs3D).shape)

print(inputs1D.shape, GAMP(inputs1D).shape)
print(inputs2D.shape, GAMP(inputs2D).shape)
print(inputs3D.shape, GAMP(inputs3D).shape)
```
</details>


#### 5.1.2 Norm&Activation
 - `customLayerNorm` a custom implementation of layer normalization
 - `MakeNorm` make normalization layer, includes BN / GN / IN / LN
 - `MakeActive` make activation layer, includes Relu / LeakyRelu
 - `MakeConv` make 1D / 2D / 3D convolutional layer

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>



#### 5.1.3 Conv
 - `ConvNormActive` 'Convolution→Normalization→Activation', used in VGG or ResNet
 - `NormActiveConv` 'Normalization→Activation→Convolution', used in DenseNet
 - `VGGBlock` the basic module in VGG
 - `VGGStage` a VGGStage = few VGGBlocks
 - `ResBlock` the basic module in ResNet
 - `ResStage` a ResStage = few ResBlocks
 - `DenseLayer` the basic module in DenseNet
 - `DenseBlock` a DenseBlock = few DenseLayers

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>

### 6.2 `wama_modules.utils`
 - `resizeTensor` scale torch tensor, similar to scipy's zoom
 - `tensor2array` transform tensor to ndarray
 - `load_weights` load torch weights and print loading details(miss keys and match keys)

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 6.3 `wama_modules.Attention`
 - `SCSEModule`
 - `NonLocal`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 5.4 `wama_modules.Encoder`
 - `VGGEncoder`
 - `ResNetEncoder`
 - `DenseNetEncoder`
 - `???`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 5.5 `wama_modules.Decoder`
 - `UNet_decoder`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 5.6 `wama_modules.Neck`
 - `FPN`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
import torch
from wama_modules.Neck import FPN

# make multi-scale feature maps
featuremaps = [
    torch.ones([3,16,32,32,32]),
    torch.ones([3,32,24,24,24]),
    torch.ones([3,64,16,16,16]),
    torch.ones([3,128,8,8,8]),
]

# build FPN
fpn_AddSmall2Big = FPN(in_channels_list=[16, 32, 64, 128],
         c1=128,
         c2=256,
         active='relu',
         norm='bn',
         gn_c=8,
         mode='AddSmall2Big',
         dim=3,)
fpn_AddBig2Small = FPN(in_channels_list=[16, 32, 64, 128],
         c1=128,
         c2=256,
         active='relu',
         norm='bn',
         gn_c=8,
         mode='AddBig2Small', # Add big size feature to small size feature, for classification
         dim=3,)

# forward
f_listA = fpn_AddSmall2Big(featuremaps)
f_listB = fpn_AddBig2Small(featuremaps)
_ = [print(i.shape) for i in featuremaps]
_ = [print(i.shape) for i in f_listA]
_ = [print(i.shape) for i in f_listB]
```
</details>


### 5.7 `wama_modules.Transformer`
 - `FeedForward`
 - `MultiHeadAttention`
 - `TransformerEncoderLayer`
 - `TransformerDecoderLayer`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


## 7. Acknowledgment 🥰
Thanks to these authors and their codes:
1) https://github.com/ZhugeKongan/torch-template-for-deep-learning
2) pytorch vit
3) SMP: https://github.com/qubvel/segmentation_models.pytorch
4) transformers
5) medicalnet
6) timm: https://github.com/rwightman/pytorch-image-models

