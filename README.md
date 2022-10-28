# ωαмα m⚙️dules 
A modular PyTorch-based neural network library, just for building 1D/2D/3D networks flexibly ~


Highlights (*Simple! Easy to use! Flexible! Feature-rich!*)
 - No complex class inheritance or nesting, and the forward process is shown succinctly
 - No complex input parameters, but output as many features as possible for fast reuse
 - No dimension restriction, 1D or 2D or 3D networks are all supported


## 1. Installation
Use the following command to install ωαмα m⚙️dules 
```
pip install git+https://github.com/WAMAWAMA/wama_modules.git
```

Or you can directly copy the `wama_modules` directory to use


## 2. How to build a network modularly?
A network can be split into few modules, such like
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


## 4. Example
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])


```

## 5. All modules (or functions)

### 5.1 `Base module`

#### Pooling
 - `GlobalAvgPool`
 - `GlobalMaxPool`
 - `GlobalMaxAvgPool`

#### Norm&Activation
 - `customLayerNorm`
 - `MakeNorm`
 - `MakeActive`
 - `MakeConv`

#### Conv
 - `ConvNormActive`
 - `VGGBlock`
 - `VGGStage`
 - `ResBlock`
 - `ResStage`


#### MLP
 - `???`
 - `???`
 - `???`
 - `???`


### 5.2 `utils`
 - `resizeTensor`
 - `tensor2array`
 - `???`
 - `???`
 - `???`
 - `???`

### 5.3 `Attention`
 - `SCSEModule`
 - `NonLocal`
 - `???`
 - `???`
 - `???`
STN?

### 5.4 `Encoder`
 - `???`



### 5.5 `Decoder`
 - `UNet_decoder`
 - `???`


### 5.6 `Neck`
 - `FPN`
 - `???`

### 5.7 `Transformer`
 - `FeedForward`
 - `MultiHeadAttention`
 - `TransformerEncoderLayer`
 - `TransformerDecoderLayer`
 - `???`




