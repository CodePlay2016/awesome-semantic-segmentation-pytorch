"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from ..nn import JPU
from .base_models.resnetv1b import resnet18_v1s, resnet18_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s
from .base_models.mobilenetv2 import mobilenet_v2_1_0, mobilenet_v2_0_25, mobilenet_v2_0_5, mobilenet_v2_0_75

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'mobilenetv2':
            self.pretrained = mobilenet_v2_1_0(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet18':
            self.pretrained = resnet18_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone == 'mobilenetv2':
            c4 = self.pretrained.features(x)
            return c4
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
