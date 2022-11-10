# ωαмα m⚙️dules


*A PyTorch module library for building 1D/2D/3D networks flexibly ~*

(*Simple-to-use & Function-rich!*)

Highlights
 - Simple code that show whole forward processes succinctly
 - Output rich features and attention map for fast reuse
 - Support 1D / 2D / 3D networks (CNNs, GNNs, Transformers...)
 - Easy and flexible to integrate with any other network
 - 🚀 Abundant Pretrained weights: Including 80000+ `2D weights` and 80+ `3D weights`



*Download pretrained weights from
[[Google Drive]](https://drive.google.com/drive/folders/1Pbgrgf8Y8soc6RGMh3bSUSWVIW2WeEmA?usp=sharing)
or [[Baidu Netdisk `psw: wama` ]](https://pan.baidu.com/s/16sskSM5IuHLbXOC4YF5MQQ?pwd=wama)

*All modules are detailed in [[Document]](Document_allmodules.md) (🚧 still under building)


## 1. Installation
🔥 [wama_modules](https://github.com/WAMAWAMA/wama_modules) 
`Basic` `1D/2D/3D`

Install `wama_modules` with command ↓
```
pip install git+https://github.com/WAMAWAMA/wama_modules.git
```
<details>
<summary> Other ways to install (or use) wama_modules</summary>

 -  Way1: Download code and run `python setup.py install`
 -  Way2: Directly copy the folder `wama_modules` into your project path

</details>

💧 [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) 
`Optional` `2D` `100+ pretrained weights`

<details>
<summary> Introduction and installation command </summary>

`segmentation_models_pytorch` (called `smp`)
is a 2D CNN lib including many backbones and decoders, which is highly recommended to install for cooperating with this library. 

*Our codes have already contained `smp`, but you can still install the latest version with the code below.

Install with pip ↓
```
pip install segmentation-models-pytorch
```
Install the latest version ↓
```
pip install git+https://github.com/rwightman/pytorch-image-models.git
```

</details>

💧 [transformers](https://github.com/huggingface/transformers) 
`Optional` `2D` `80000+ pretrained weights`

<details>
<summary> Introduction and installation command </summary>

`transformer` (powered by Huggingface) is a lib including super abundant CNN and Transformer structures, which is highly recommended to install for cooperating with this library. 

Install `transformer` with pip ↓
```
pip install transformers
```

</details>

💧 [timm](https://github.com/rwightman/pytorch-image-models) 
`Optional` `2D` `400+ pretrained weights` 

<details>
<summary> Introduction and installation command </summary>

`timm` is a lib includes abundant CNN and Transformer structures, which is highly recommended to install for cooperating with this library. 

Install with pip ↓
```
pip install timm
```
Install the latest version ↓
```
pip install git+https://github.com/rwightman/pytorch-image-models.git
```
</details>


## 2. Update list
 - 2022/11/11:  The birthday of this code, version `v0.0.1`
 - ...


## 3. Main modules and network architectures
An overview of this repo (let's call `wama_modules` as `wm`)


|File|Description  | Main class or function  |
|---|---|---|
|`wm.utils`          |Some operations on tensors and pre-training weights | `resizeTensor()` `tensor2array()` `load_weights()` |
|`wm.thirdparty_lib` |2D/3D network structures (CNN/GNN/Transformer) from other repositories, and all are with pre-trained weights 🚀  | `MedicalNet` `C3D` `3D_DenseNet` `3D_shufflenet` `transformers.ConvNextModel` `transformers.SwinModel` `Radimagenet`|
|`wm.Attention`      |Some attention-based plugins   | `SCSEModule` `NonLocal`  |
|`wm.BaseModule`     |Basic modules(layers). For example, BottleNeck block (ResBlock) in ResNet, and DenseBlock in DenseNet, etc.   |`MakeNorm()` `MakeConv()` `MakeActive()` `VGGBlock` `ResBlock` `DenseBlock` |
|`wm.Encoder`        |Some encoders such like ResNet or DenseNet, but with more flexibility for building the network modularly, and 1D/2D/3D are all supported   |`VGGEncoder` `ResNetEncoder` `DenseNetEncoder` |
|`wm.Decoder`        |Some encoders with more flexibility for building the network modularly, and 1D/2D/3D are all supported   | `UNet_decoder`  |
|`wm.Neck`           |Modules for making the multi-scale features (from encoder) interact with each other to generate stronger features | `FPN` |
|`wm.Transformer`    |Some self-attention or cross-attention modules, which can be used to build ViT, DETR or TransUnet   | `TransformerEncoderLayer` `TransformerDecoderLayer` |


 - How to build your networks modularly and freely? 👉 See ['Guideline 1: Build networks modularly'](https://github.com/WAMAWAMA/wama_modules#4-guideline-1-build-networks-modularly)  below ~
 - How to use pretrained model with `wm.thirdparty_lib`? 👉 See ['Guideline 2: Use pretrained weights'](https://github.com/WAMAWAMA/wama_modules#5-guideline-2-use-pretrained-weights)  below ~




## 4. Guideline 1: Build networks modularly
How to build a network modularly?  Here's a paradigm:

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






*Todo-demo list (under preparation and coming soon...) ↓

<details>
<summary> Demo1: Build a 2D vgg16  </summary>
 
```python
```
</details>

<details>
<summary> Demo2: Build a 3D resnet50  </summary>
 
```python
```
</details>



<details>
<summary> Demo3: Build a 3D densenet121  </summary>
 
```python
```
</details>


<details>
<summary> Demo4: Build a Unet  </summary>
 
```python
```
</details>


<details>
<summary> Demo5: Build a Unet with a resnet50 encoder  </summary>
 
```python
```
</details>

<details>
<summary> Demo6: Build a Unet with a resnet50 encoder and a FPN </summary>
 
```python
```
</details>

<details>
<summary> Demo7: Build a multi-task model for segmentation and classification</summary>
 
```python
```
</details>



<details>
<summary> Demo8: Build a C-tran model for multi-label classification</summary>
 
```python
```
</details>


<details>
<summary> Demo9: Build a Q2L model for multi-label classification</summary>
 
```python
```
</details>

<details>
<summary> Demo10: Build a ML-Decoder model for multi-label classification</summary>
 
```python
```
</details>


<details>
<summary> Demo11: Build a ML-GCN model for multi-label classification</summary>
 
```python
```
</details>


<details>
<summary> Demo12: Build a UCTransNet model for segmentation </summary>
 
```python
```
</details>

<details>
<summary> Demo13: Build a model for multiple inputs (1D signal and 2D image) </summary>

```python
```
</details>


<details>
<summary> Demo14: Build a 2D Unet with pretrained Resnet50 encoder (1D signal and 2D image) </summary>
 
```python
```
</details>


<details>
<summary> Demo15: Build a 3D DETR model for object detection </summary>
 
```python
```
</details>

<details>
<summary> Demo16: Build a 3D VGG with SE-attention module for multi-instanse classification </summary>
 
```python
```
</details>



## 5. Guideline 2: Use pretrained weights 

(*All pretrained weights are from third-party codes or repos)

Currently available pre-training models are shown below ↓

| |Module name| Number of pretrained weights | Pretrained data  | Dimension   |
|---|---|---|---|---|
| 1 | `.ResNets3D_kenshohara` | 21 | video| 3D |
| 2 | `.VC3D_kenshohara` | 13 | video| 3D |
| 3 | `.Efficient3D_okankop` | 39 | video|3D |
| 4 | `.MedicalNet_tencent` | 11 | medical image|3D |
| 5 | `.C3D_jfzhang95` | 1 | video| 3D|
| 6 | `.C3D_yyuanad` | 1 | video| 3D|
| 7 | `.SMP_qubvel` | 119 | image|2D |
| 8 | `timm` | 400+ | image| 2D|
| 9 | `transformers` | 80000+ | video/image| 2D/3D|
| 10| `radimagenet` | 1 | medical image| 2D|



*Download all pretrained weights from
[[Google Drive]](https://drive.google.com/drive/folders/1Pbgrgf8Y8soc6RGMh3bSUSWVIW2WeEmA?usp=sharing)
or [[Baidu Netdisk `psw: wama` ]](https://pan.baidu.com/s/16sskSM5IuHLbXOC4YF5MQQ?pwd=wama)

### 5.1  ResNets3D_kenshohara  `21 weights` `3D` 

ResNets3D_kenshohara (21 weights)

<details>
<summary> Demo code --------------------------------- </summary>
 
```python
import torch
from wama_modules.thirdparty_lib.ResNets3D_kenshohara.resnet import generate_model
from wama_modules.utils import load_weights
m = generate_model(18)
pretrain_path = r"D:\pretrainedweights\ResNets3D_kenshohara\kenshohara_ResNets3D_weights\resnet\r3d18_KM_200ep.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights)
f_list = m(torch.ones([2,3,64,64,64]))
_ = [print(i.shape) for i in f_list]


import torch
from wama_modules.thirdparty_lib.ResNets3D_kenshohara.resnet2p1d import generate_model
from wama_modules.utils import load_weights
m = generate_model(18)
pretrain_path = r"D:\pretrainedweights\ResNets3D_kenshohara\kenshohara_ResNets3D_weights\resnet2p1d\r2p1d18_K_200ep.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights)
f_list = m(torch.ones([2,3,64,64,64]))
_ = [print(i.shape) for i in f_list]
```
</details>

### 5.2  VC3D_kenshohara  `13 weights` `3D`

VC3D_kenshohara (13 weights)

<details>
<summary> Demo code --------------------------------- </summary>

```python
# resnet
import torch
from wama_modules.thirdparty_lib.VC3D_kenshohara.resnet import generate_model
from wama_modules.utils import load_weights
m = generate_model(18)
pretrain_path = r"D:\pretrainedweights\VC3D_kenshohara\VC3D_weights\resnet\resnet-18-kinetics.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2,3,64,64,64]))
_ = [print(i.shape) for i in f_list]

# resnext
import torch
from wama_modules.thirdparty_lib.VC3D_kenshohara.resnext import generate_model
from wama_modules.utils import load_weights
m = generate_model(101)
pretrain_path = r"D:\pretrainedweights\VC3D_kenshohara\VC3D_weights\resnext\resnext-101-64f-kinetics.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2,3,64,64,64]))
_ = [print(i.shape) for i in f_list]

# wide_resnet
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

</details>


### 5.3  Efficient3D_okankop  `39 weights` `3D`

Efficient3D_okankop (39 weights)

<details>
<summary> Demo code --------------------------------- </summary>

```python
# c3d
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.c3d import get_model
m = get_model()  # c3d has no pretrained weights
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# mobilenet
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.mobilenet import get_model
from wama_modules.utils import load_weights
m = get_model(width_mult = 1.)  # e.g. width_mult = 1 when mobilenet_1.0x
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\mobilenet\jester_mobilenet_1.0x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]

m = get_model(width_mult = 2.)  # e.g. width_mult = 2 when mobilenet_2.0x
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\mobilenet\jester_mobilenet_2.0x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# mobilenetv2
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.mobilenetv2 import get_model
from wama_modules.utils import load_weights
m = get_model(width_mult = 1.)  # e.g. width_mult = 1 when mobilenet_1.0x
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\mobilenetv2\jester_mobilenetv2_1.0x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


m = get_model(width_mult = 0.45)  # e.g. width_mult = 1 when mobilenet_1.0x
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\mobilenetv2\jester_mobilenetv2_0.45x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# resnet
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.resnet import resnet18, resnet50, resnet101
from wama_modules.utils import load_weights
m = resnet18()
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\resnet\kinetics_resnet_18_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]

m = resnet50()
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\resnet\kinetics_resnet_50_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]

m = resnet101()
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\resnet\kinetics_resnet_101_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# resnext
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.resnext import resnext101
from wama_modules.utils import load_weights
m = resnext101()
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\resnext\jester_resnext_101_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# shufflenet
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.shufflenet import get_model
from wama_modules.utils import load_weights
m = get_model(groups=3, width_mult=1)
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\shufflenet\jester_shufflenet_1.0x_G3_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]

m = get_model(groups=3, width_mult=1.5)
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\shufflenet\jester_shufflenet_1.5x_G3_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# shufflenetv2
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.shufflenetv2 import get_model
from wama_modules.utils import load_weights
m = get_model(width_mult=1)
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\shufflenetv2\jester_shufflenetv2_1.0x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]

m = get_model(width_mult=2)
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\shufflenetv2\jester_shufflenetv2_2.0x_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]


# squeezenet
import torch
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.squeezenet import get_model
from wama_modules.utils import load_weights
m = get_model()
pretrain_path = r"D:\pretrainedweights\Efficient3D_okankop\Efficient3D_okankop_weights\squeezenet\jester_squeezenet_RGB_16_best.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]
```

</details>



### 5.4  MedicalNet_tencent  `11 weights` `3D` `medical`

MedicalNet_tencent (11 weights)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.MedicalNet_Tencent.model import generate_model
m = generate_model(18)
pretrain_path = r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
m = load_weights(m, pretrain_weights, drop_modelDOT=True)
f_list = m(torch.ones([2, 1, 64, 64, 64]))  # input channel is 1 (not 3 for video)
_ = [print(i.shape) for i in f_list]
```

</details>


### 5.5  C3D_jfzhang95  `1 weights` `3D`

C3D_jfzhang95 (1 weight)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.C3D_jfzhang95.c3d import C3D
m = C3D()
pretrain_path = r"D:\pretrainedweights\C3D_jfzhang95\C3D_jfzhang95_weights\C3D_jfzhang95_C3D.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')
m = load_weights(m, pretrain_weights)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]
```

</details>


### 5.6  C3D_yyuanad  `1 weights` `3D`

C3D_yyuanad (1 weight)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.C3D_yyuanad.c3d import C3D
m = C3D()
pretrain_path = r"D:\pretrainedweights\C3D_yyuanad\C3D_yyuanad_weights\C3D_yyuanad.pickle"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')
m = load_weights(m, pretrain_weights)
f_list = m(torch.ones([2, 3, 64, 64, 64]))
_ = [print(i.shape) for i in f_list]
```

</details>


### 5.7  SMP_qubvel  `119 weights` `2D`

SMP_qubvel (119 weight)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ssl')
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]
```

</details>



### 5.8  timm  `400+ weights` `2D`

timm (400+ weight)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
import timm
m = timm.create_model(
    'adv_inception_v3',
    features_only=True,
    pretrained=True,)
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]
```

</details>


### 5.9  transformers  `80000+ weights` `2D`

transformers (80000+ weight), all models please go to Huggingface [[ModelHub]](https://huggingface.co/models)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from transformers import ConvNextModel
from wama_modules.utils import load_weights
# Initializing a model (with random weights) from the convnext-tiny-224 style configuration
m = ConvNextModel.from_pretrained('facebook/convnext-base-224-22k')
f = m(torch.ones([2,3,224,224]), output_hidden_states=True)
f_list = f.hidden_states
_ = [print(i.shape) for i in f_list]
# reload weights
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
# reload weights
weights = m.state_dict()
m1 = SwinModel(m.config)
m = load_weights(m, weights)
```

</details>



### 5.10  radimagenet  `1 weights` `2D` `medical`

radimagenet (1 weight)

<details>
<summary> Demo code --------------------------------- </summary>

```python
import torch
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
m = get_encoder('resnet50', in_channels=3, depth=5, weights=None)
pretrain_path = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-ResNet50_notop_torch.pth"
pretrain_weights = torch.load(pretrain_path, map_location='cpu')
m = load_weights(m, pretrain_weights)
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]

```

</details>















## 6. Acknowledgment 🥰
Thanks to these authors and their codes:
1) https://github.com/ZhugeKongan/torch-template-for-deep-learning
2) pytorch vit: https://github.com/lucidrains/vit-pytorch
3) SMP: https://github.com/qubvel/segmentation_models.pytorch
4) transformers: https://github.com/huggingface/transformers
5) medicalnet: https://github.com/Tencent/MedicalNet
6) timm: https://github.com/rwightman/pytorch-image-models
7) ResNets3D_kenshohara: https://github.com/kenshohara/3D-ResNets-PyTorch
8) VC3D_kenshohara: https://github.com/kenshohara/video-classification-3d-cnn-pytorch
9) Efficient3D_okankop: https://github.com/okankop/Efficient-3DCNNs
10) C3D_jfzhang95: https://github.com/jfzhang95/pytorch-video-recognition
11) C3D_yyuanad: https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor
12) radimagenet: https://github.com/BMEII-AI/RadImageNet
13) https://github.com/BMEII-AI/RadImageNet/issues/3

