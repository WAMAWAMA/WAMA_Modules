# ωαмα m⚙️dules
A PyTorch-based module library for building 1D/2D/3D networks flexibly ~


Highlights (*Simple-to-use & Function-rich!*)
 - No complex class inheritance or nesting, and the forward process is shown succinctly
 - No complex input parameters, but output as many features as possible for fast reuse
 - No dimension restriction, 1D or 2D or 3D networks are all supported


## 1. Installation
Use the following command to install `wama_modules` 
```
pip install git+https://github.com/WAMAWAMA/wama_modules.git
```

Or you can directly copy the `wama_modules` directory to use

---
Optional: Highly recommended to install [`smp`](https://github.com/qubvel/segmentation_models.pytorch), 
which contains many 2D networks that can be used with this  `wama_modules` library 

Use the following command to install [`smp`](https://github.com/qubvel/segmentation_models.pytorch)
```
pip install git+https://github.com/qubvel/segmentation_models.pytorch
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


### 5.2 `wama_modules.utils`
 - `resizeTensor`
 - `tensor2array`


### 5.3 `wama_modules.Attention`
 - `SCSEModule`
 - `NonLocal`


### 5.4 `wama_modules.Encoder`
 - `???`


### 5.5 `wama_modules.Decoder`
 - `UNet_decoder`


### 5.6 `wama_modules.Neck`
 - `FPN`


### 5.7 `wama_modules.Transformer`
 - `FeedForward`
 - `MultiHeadAttention`
 - `TransformerEncoderLayer`
 - `TransformerDecoderLayer`


## 6. Acknowledgment
Thanks to ......