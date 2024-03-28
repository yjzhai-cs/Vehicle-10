import torchvision
from torch import nn
from .lenet5 import LeNet5
from .resnet9 import ResNet9

def get_model(name):
    if name == 'resnet9':
        net = ResNet9(in_channels=3, num_classes=10)
    elif name == 'resnet18':
        net = torchvision.models.resnet18(num_classes=10)
    elif name == 'resnet34':
        net = torchvision.models.resnet34(num_classes=10)
    elif name == 'resnet50':
        net = torchvision.models.resnet50(num_classes=10)
    elif name == 'vgg16':
        net = torchvision.models.vgg16(num_classes=10)
    elif name == 'vgg19':
        net = torchvision.models.vgg19(num_classes=10)
    elif name == 'lenet5':
        net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
    else:
        raise RuntimeError(f'{name} model is undefined')
    return net