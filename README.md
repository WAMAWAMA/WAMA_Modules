# ωαмα m⚙️dules
A PyTorch-based module library for building 1D/2D/3D networks flexibly ~


Highlights (*Simple-to-use & Function-rich!*)
 - No complex class inheritance or nesting, and the forward process is shown succinctly
 - No complex input parameters, but output as many features as possible for fast reuse
 - No dimension restriction, 1D or 2D or 3D networks are all supported


## 1. Installation
 - 🔥1.1 [`wama_modules`](https://github.com/WAMAWAMA/wama_modules) (*Basic*) 
   
Install *wama_modules* use ↓
```
pip install git+https://github.com/WAMAWAMA/wama_modules.git
```

 Or you can directly copy the *wama_modules* folder to use


 - 💧1.2 [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) (*Optional*) 

*segmentation_models_pytorch* (called *smp*)
is a 2D CNN lib includes many backbones and decoders, which is highly recommended to install for cooperating with this library. 
Install *smp* use ↓
```
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

 - 💧1.3 [`transformers`](https://github.com/huggingface/transformers) (*Optional*) 
   
*transformer* is a lib includes abundant Transformer structures, which is highly recommended to install for cooperating with this library. 
Install *transformer* use ↓
```
pip install transformers
```



## 2. How to build a network modularly?

Building a network must follow the following paradigm:

***'Design the architecture according to the tasks, and pick the right modules for the designed architecture'***

So, networks for different tasks can be designed modularly such as:
 - vgg = vgg_encoder + cls_head
 - Unet = encoder + decoder + seg_ead
 - resnet = resnet_encoder + cls_head
 - densenet = densenet_encoder + cls_head
 - a multi-task net for classification and segmentation = encoder + decoder + cls_head + seg_head


## 3. Main modules
 - resblock？
 - dense block
 - decoder block
 - transformer block


## 4. Examples


Build a 3D resnet50 


```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])

```





More demos are shown below ↓ (Click to view codes), or you can visit the `demo` folder for more demo codes



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



## 5. All modules (or functions)

### 5.1 `wama_modules.BaseModule`

#### 5.1.1 Pooling
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

### 5.2 `wama_modules.utils`
 - `resizeTensor` scale torch tensor, similar to scipy's zoom
 - `tensor2array` transform tensor to ndarray

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 5.3 `wama_modules.Attention`
 - `SCSEModule`
 - `NonLocal`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


### 5.4 `wama_modules.Encoder`
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


## 6. Acknowledgment
Thanks to ......