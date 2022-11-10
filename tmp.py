
# todo 1 3D ResNets3D_kenshohara (21 weights)
if True:
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
if True:
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


# todo 4 3D MedicalNet_tencent (11 weights)
if True:
    import torch
    from wama_modules.utils import load_weights
    from wama_modules.thirdparty_lib.MedicalNet_Tencent.model import generate_model
    m = generate_model(18)
    pretrain_path = r"D:\pretrainedweights\MedicalNet_Tencent\MedicalNet_weights\resnet_18_23dataset.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')['state_dict']
    m = load_weights(m, pretrain_weights, drop_modelDOT=True)
    f_list = m(torch.ones([2, 1, 64, 64, 64]))  # input channel is 1 (not 3 for video)
    _ = [print(i.shape) for i in f_list]


# todo 5 3D C3D_jfzhang95 (1 weight)
if True:
    import torch
    from wama_modules.utils import load_weights
    from wama_modules.thirdparty_lib.C3D_jfzhang95.c3d import C3D
    m = C3D()
    pretrain_path = r"D:\pretrainedweights\C3D_jfzhang95\C3D_jfzhang95_weights\C3D_jfzhang95_C3D.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')
    m = load_weights(m, pretrain_weights)
    f_list = m(torch.ones([2, 3, 64, 64, 64]))
    _ = [print(i.shape) for i in f_list]


# todo 6 3D C3D_yyuanad (1 weight)
if True:
    import torch
    from wama_modules.utils import load_weights
    from wama_modules.thirdparty_lib.C3D_yyuanad.c3d import C3D
    m = C3D()
    pretrain_path = r"D:\pretrainedweights\C3D_yyuanad\C3D_yyuanad_weights\C3D_yyuanad.pickle"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')
    m = load_weights(m, pretrain_weights)
    f_list = m(torch.ones([2, 3, 64, 64, 64]))
    _ = [print(i.shape) for i in f_list]


# todo 7 2D smp (119 weight)
if True:
    # smp
    import torch
    from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
    m = get_encoder('resnet18', in_channels=3, depth=5, weights='ssl')
    f_list = m(torch.ones([2,3,128,128]))
    _ = [print(i.shape) for i in f_list]


# todo 8 timm (400+)
if True:
    import timm
    m = timm.create_model(
        'adv_inception_v3',
        features_only=True,
        pretrained=True,)
    f_list = m(torch.ones([2,3,128,128]))
    _ = [print(i.shape) for i in f_list]


# todo 9 transformers (80000+ weights)
if True:
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


# todo 10 radimagenet (4 weights)
if True:
    import torch
    from wama_modules.utils import load_weights
    from wama_modules.thirdparty_lib.SMP_qubvel.encoders import get_encoder
    m = get_encoder('resnet50', in_channels=3, depth=5, weights=None)
    pretrain_path = r"D:\pretrainedweights\radimagnet\RadImageNet_models-20221104T172755Z-001\RadImageNet_models\RadImageNet-ResNet50_notop_torch.pth"
    pretrain_weights = torch.load(pretrain_path, map_location='cpu')
    m = load_weights(m, pretrain_weights)
    f_list = m(torch.ones([2,3,128,128]))
    _ = [print(i.shape) for i in f_list]





