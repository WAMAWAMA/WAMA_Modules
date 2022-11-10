
# All modules and functions

## 1 `wama_modules.BaseModule`

### 1.1 Pooling
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


### 1.2 Norm&Activation
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



### 1.3 Conv
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

## 2 `wama_modules.utils`
 - `resizeTensor` scale torch tensor, similar to scipy's zoom
 - `tensor2array` transform tensor to ndarray
 - `load_weights` load torch weights and print loading details(miss keys and match keys)

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


## 3 `wama_modules.Attention`
 - `SCSEModule`
 - `NonLocal`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


## 4 `wama_modules.Encoder`
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


## 5 `wama_modules.Decoder`
 - `UNet_decoder`

<details>
<summary> Click here to see demo code </summary>
 
```python
""" demo """
```
</details>


## 6 `wama_modules.Neck`
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


## 7 `wama_modules.Transformer`
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
