
# todo 1 3D ResNets3D_kenshohara (21 weights)
if True:
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


# todo 2 3D VC3D_kenshohara (13 weights)
if True:
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


# todo 3 3D Efficient3D_okankop (39 weights)



# todo 4 3D MedicalNet_tencent  (11 weights)



# todo 5 3D C3D_jfzhang95 (1 weight)



# todo 6 3D C3D_yyuanad (1 weight)



# todo 7 2D smp (119 weight)
# smp
import torch
from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ssl')
m = get_encoder('name', in_channels=3, depth=5, weights='ssl')
m = get_encoder('resnet18', in_channels=3, depth=5, weights='ss')
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]


# todo 8 timm (400+)
import timm
m = timm.create_model(
    'adv_inception_v3',
    features_only=True,
    pretrained=False,)
f_list = m(torch.ones([2,3,128,128]))
_ = [print(i.shape) for i in f_list]



# todo 9 transformers (80000+ weights)
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








