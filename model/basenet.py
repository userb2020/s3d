from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from utils.utils import gap2d, var2d
from torch.distributions import Beta

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def forward_mean_var(self, x, sty_layer):
        x = self.features[0](x)  # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        x = self.features[1](x)
        x = self.features[2](x)
        x1_cm = gap2d(x, keepdims=True)
        x1_cv = var2d(x, keepdims=True)

        x = self.features[3](x)  # nn.Conv2d(64, 192, kernel_size=5, padding=2)
        x = self.features[4](x)
        x = self.features[5](x)
        if sty_layer in ['layer2', 'layer3', 'layer4']:
            x2_cm = gap2d(x, keepdims=True)
            x2_cv = var2d(x, keepdims=True)
        else:
            x2_cm = 0
            x2_cv = 0

        x = self.features[6](x)  # nn.Conv2d(192, 384, kernel_size=3, padding=1)
        x = self.features[7](x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = gap2d(x, keepdims=True)
            x3_cv = var2d(x, keepdims=True)
        else:
            x3_cm = 0
            x3_cv = 0
        x = self.features[8](x)  # nn.Conv2d(384, 256, kernel_size=3, padding=1)
        x = self.features[9](x)
        if sty_layer in ['layer4']:
            x4_cm = gap2d(x, keepdims=True)
            x4_cv = var2d(x, keepdims=True)
        else:
            x4_cm = 0
            x4_cv = 0

        x = self.features[10](x)  # nn.Conv2d(256, 256, kernel_size=3, padding=1)
        x = self.features[11](x)
        x = self.features[12](x)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        x_sty = [(x1_cm, x1_cv), (x2_cm, x2_cv), (x3_cm, x3_cv), (x4_cm, x4_cv)]
        return x, x_sty

    def forward_assistant(self, x, x_sty, sty_w, sty_layer):
        x = self.features[0](x)  # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        x = self.features[1](x)
        x = self.features[2](x)
        x1_cm = gap2d(x, keepdims=True)
        x1_cv = var2d(x, keepdims=True)
        sty_x = self.styletransform_detach(x, x1_cm, x1_cv, x_sty[0][0], x_sty[0][1], sty_w)

        x = self.features[3](x)  # nn.Conv2d(64, 192, kernel_size=5, padding=2)
        x = self.features[4](x)
        x = self.features[5](x)
        sty_x = self.features[3](sty_x)
        sty_x = self.features[4](sty_x)
        sty_x = self.features[5](sty_x)
        if sty_layer in ['layer2', 'layer3','layer4']:
            x2_cm = gap2d(x, keepdims=True)
            x2_cv = var2d(x, keepdims=True)
            sty_x = self.styletransform_detach(sty_x, x2_cm, x2_cv, x_sty[1][0], x_sty[1][1], sty_w)

        x = self.features[6](x)  # nn.Conv2d(192, 384, kernel_size=3, padding=1)
        x = self.features[7](x)
        sty_x = self.features[6](sty_x)
        sty_x = self.features[7](sty_x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = gap2d(x, keepdims=True)
            x3_cv = var2d(x, keepdims=True)
            sty_x = self.styletransform_detach(sty_x, x3_cm, x3_cv, x_sty[2][0], x_sty[2][1], sty_w)

        x = self.features[8](x)  # nn.Conv2d(384, 256, kernel_size=3, padding=1)
        x = self.features[9](x)
        sty_x = self.features[8](sty_x)
        sty_x = self.features[9](sty_x)
        if sty_layer in ['layer4']:
            x4_cm = gap2d(x, keepdims=True)
            x4_cv = var2d(x, keepdims=True)
            sty_x = self.styletransform_detach(sty_x, x4_cm, x4_cv, x_sty[3][0], x_sty[3][1], sty_w)

        x = self.features[10](x)  # nn.Conv2d(256, 256, kernel_size=3, padding=1)
        x = self.features[11](x)
        x = self.features[12](x)
        sty_x = self.features[10](sty_x)
        sty_x = self.features[11](sty_x)
        sty_x = self.features[12](sty_x)

        x = x.view(x.size(0), 256 * 6 * 6)
        sty_x = sty_x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        sty_x = self.classifier(sty_x)

        return x, sty_x

    def output_num(self):
        return self.__in_features


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


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out
