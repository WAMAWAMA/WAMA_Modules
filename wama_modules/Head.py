import torch
from wama_modules.BaseModule import *


class ClassificationHead(nn.Module):
    """
    Head for single or multiple label classification task
    """

    def __init__(self, label_category_dict, in_channel, bias=True):
        super().__init__()
        self.classification_head = torch.nn.ModuleDict({})
        for key in label_category_dict.keys():
            self.classification_head[key] = torch.nn.Linear(in_channel, label_category_dict[key], bias=bias)

    def forward(self, f):
        """
        # demo: an example of fruit classification task

        f = torch.ones([3, 512])  # from encoder
        label_category_dict = dict(
            shape=4,
            color=3,
            rotten=2,
            sweet=2,
            sour=2,
        )
        cls_head = ClassificationHead(label_category_dict, 512)
        logits = cls_head(f)
        _ = [print('logits of ', key,':' ,logits[key].shape) for key in logits.keys()]


        # support element-wise performing, by transfer dict or list to f
        label_category_dict = dict(
            shape=4,
            color=3,
            rotten=2,
            sweet=2,
            sour=2,
        )

        # dict input for element-wised FC
        f = {}
        for key in label_category_dict.keys():
            f[key] = torch.ones([3, 512])  # from encoder
        cls_head = ClassificationHead(label_category_dict, 512)
        logits = cls_head(f)
        _ = [print('logits of ', key,':' ,logits[key].shape) for key in logits.keys()]

        # list input for element-wised FC
        f = []
        for key in label_category_dict.keys():
            f.append(torch.ones([3, 512]))  # from encoder
        cls_head = ClassificationHead(label_category_dict, 512)
        logits = cls_head(f)
        _ = [print('logits of ', key,':' ,logits[key].shape) for key in logits.keys()]

        """
        logits = {}
        if isinstance(f,dict):
            print('dict element-wised forward')
            for key in self.classification_head.keys():
                logits[key] = self.classification_head[key](f[key])
        elif isinstance(f,list):
            print('list element-wised forward')
            for key_index, key in enumerate(self.classification_head.keys()):
                logits[key] = self.classification_head[key](f[key_index])
        else:
            for key in self.classification_head.keys():
                logits[key] = self.classification_head[key](f)
        return logits


class SegmentationHead(nn.Module):
    """Head for single or multiple label segmentation task"""

    def __init__(self, label_category_dict, in_channel, bias=True, dim=2):
        super().__init__()
        self.segmentatin_head = torch.nn.ModuleDict({})
        for key in label_category_dict.keys():
            self.segmentatin_head[key] = MakeConv(in_channel, label_category_dict[key], 3, padding=1, stride=1, dim=dim,
                                                  bias=bias)

    def forward(self, f):
        """
        # demo 2D

        f = torch.ones([3, 512, 128, 128])  # from decoder or encoder
        label_category_dict = dict(
            organ=14, # 14 kinds of organ
            tumor=3, # 3 kinds of tumor
        )
        seg_head = SegmentationHead(label_category_dict, 512, dim=2)
        seg_logits = seg_head(f)
        _ = [print('segmentation_logits of ', key,':' ,seg_logits[key].shape) for key in seg_logits.keys()]

        """
        logits = {}
        for key in self.segmentatin_head.keys():
            logits[key] = self.segmentatin_head[key](f)
        return logits
