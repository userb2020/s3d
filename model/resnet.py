from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Function
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from utils.utils import gap2d, var2d
from torch.distributions import Beta

__all__ = ['ResNet', 'resnet34']

model_urls = {
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or \
       classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        print(self.scale)
        return input * self.scale


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.nobn = nobn

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    # for extract mean and variance of labeled batch
    def forward_mean_var(self, x, sty_layer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1_cm = gap2d(x, keepdims=True)
        x1_cv = var2d(x, keepdims=True)

        x = self.layer2(x)
        if sty_layer in ['layer2', 'layer3', 'layer4']:
            x2_cm = gap2d(x, keepdims=True)
            x2_cv = var2d(x, keepdims=True)
        else:
            x2_cm = 0
            x2_cv = 0

        x = self.layer3(x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = gap2d(x, keepdims=True)
            x3_cv = var2d(x, keepdims=True)
        else:
            x3_cm = 0
            x3_cv = 0

        x = self.layer4(x)
        if sty_layer in ['layer4']:
            x4_cm = gap2d(x, keepdims=True)
            x4_cv = var2d(x, keepdims=True)
        else:
            x4_cm = 0
            x4_cv = 0

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_sty = [(x1_cm, x1_cv), (x2_cm, x2_cv), (x3_cm, x3_cv), (x4_cm, x4_cv)]
        # x1_cm [batch x 64 x 1 x 1]
        # x2_cm [batch x 128 x 1 x 1]
        # x3_cm [batch x 256 x 1 x 1]
        # x4_cm [batch x 512 x 1 x 1]

        return x, x_sty

    # for generating assistant for unlabeled batch
    # extract ordinary unl_feature and assistant
    # x_sty is from labeled batch
    def forward_assistant(self, x, x_sty, sty_w, sty_layer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1_cm = gap2d(x, keepdims=True)
        x1_cv = var2d(x, keepdims=True)
        assist_x = self.styletransform_detach(x, x1_cm, x1_cv, x_sty[0][0], x_sty[0][1], sty_w)

        x = self.layer2(x)
        assist_x = self.layer2(assist_x)
        if sty_layer in ['layer2', 'layer3', 'layer4']:
            x2_cm = gap2d(assist_x, keepdims=True)
            x2_cv = var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x2_cm, x2_cv, x_sty[1][0], x_sty[1][1], sty_w)

        x = self.layer3(x)
        assist_x = self.layer3(assist_x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = gap2d(assist_x, keepdims=True)
            x3_cv = var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x3_cm, x3_cv, x_sty[2][0], x_sty[2][1], sty_w)

        x = self.layer4(x)
        assist_x = self.layer4(assist_x)
        if sty_layer in ['layer4']:
            x4_cm = gap2d(assist_x, keepdims=True)
            x4_cv = var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x4_cm, x4_cv, x_sty[3][0], x_sty[3][1], sty_w)

        x = self.avgpool(x)
        assist_x = self.avgpool(assist_x)
        x = x.view(x.size(0), -1)
        assist_x = assist_x.view(assist_x.size(0), -1)
        return x, assist_x


    def styletransform_detach(self, x, x_m, x_v, y_m, y_v, sty_w):
        x_m, x_v, y_m, y_v = x_m.detach(), x_v.detach(), y_m.detach(), y_v.detach()
        eps = 1e-6

        batch_size = x.size(0)

        lmda = Beta(sty_w, sty_w).sample((batch_size, 1, 1, 1))
        lmda = lmda.cuda()

        # variance to standard deviation
        x_v = (x_v + eps).sqrt()
        y_v = (y_v + eps).sqrt()

        sty_mean = lmda * y_m + (1 - lmda) * x_m
        sty_std = lmda * y_v + (1 - lmda) * x_v


        sty_x = sty_std * ((x - x_m) / x_v) + sty_mean

        return sty_x


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
