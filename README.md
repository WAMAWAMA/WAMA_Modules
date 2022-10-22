# ωαмα m⚙️dules 
A modular PyTorch-based neural network library, just for building 1D/2D/3D/ND networks flexibly ~

A net work can be split into few modules, such like
 - vgg = encoder + cls head
 - Unet
 - resnet
 - densenet
 - deeplabv3


# Main modules
 - resblock？
 - dense block
 - decoder block
 - transformer block


# Example
```python
import wama_modules as ws
import torch

encoder = ws.resnet(input_channel = 3, per_stage_channel = [8,16,32,64], dim=3)
decoder = ws.unet(encoder = encoder, output_channel = 3, dim=3)

input = torch.ones([3,3,128,128])


```

# 3 All modules (or functions)

## 3.1 `Base module`

### Pooling
 - `GlobalAvgPool`
 - `GlobalMaxPool`
 - `GlobalMaxAvgPool`

### Norm&Activation
 - `customLayerNorm`
 - `MakeNorm`
 - `MakeActive`
 - `MakeConv`


### Conv
 - `ConvNormActive`



### MLP
 - `xxx`


## 3.2 `utils`
 - `resizeTensor`
 - `tensor2array`

## 3.3 `Attention`
 - `xxx`
 - `xxx`




