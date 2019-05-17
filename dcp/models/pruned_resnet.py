# copy from pytorch-torchvision-models-resnet
import math

import torch.nn as nn

__all__ = ['PrunedResNet', 'PrunedBasicBlock', 'PrunedBottleneck']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PrunedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, pruning_rate, stride=1, downsample=None):
        super(PrunedBasicBlock, self).__init__()
        self.name = "resnet-basic"
        self.pruned_channel_plane = int(planes - math.floor(planes * pruning_rate))

        self.conv1 = conv3x3(inplanes, self.pruned_channel_plane, stride)
        self.bn1 = nn.BatchNorm2d(self.pruned_channel_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.pruned_channel_plane, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
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


class PrunedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, pruning_rate, stride=1, downsample=None):
        super(PrunedBottleneck, self).__init__()
        self.name = "resnet-bottleneck"
        self.pruned_channel_plane = int(planes - math.floor(planes * pruning_rate))

        self.conv1 = nn.Conv2d(inplanes, self.pruned_channel_plane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.pruned_channel_plane)
        self.conv2 = nn.Conv2d(self.pruned_channel_plane, self.pruned_channel_plane, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.pruned_channel_plane)
        self.conv3 = nn.Conv2d(self.pruned_channel_plane, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

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


class PrunedResNet(nn.Module):

    def __init__(self, depth, pruning_rate, num_classes=1000):
        self.inplanes = 64
        super(PrunedResNet, self).__init__()
        if depth < 50:
            block = PrunedBasicBlock
        else:
            block = PrunedBottleneck

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], pruning_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], pruning_rate, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], pruning_rate, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], pruning_rate, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, pruning_rate, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pruning_rate, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, pruning_rate))

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
        x = self.fc(x)

        return x