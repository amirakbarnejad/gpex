'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class AkClamp(torch.nn.Module):
    def __init_(self, minval=-1.0, maxval=1.0):
        self.minval, self.maxval = minval, maxval
        super(AkClamp, self).__init__()
        
        
    def forward(self, x):
        output = torch.clamp(
                    x, -1.0, 1.0
                  )
        return output

class ModuleLinearView(nn.Module):
    def __init__(self):
       super(ModuleLinearView, self).__init__() 
       
    def forward(self, x):
        output = x.view(x.size(0), -1)
        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dim_wideoutput, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.dim_wideoutput = dim_wideoutput
        self.num_blocks = num_blocks
        conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layers_conv1bn1relu_1234_avgpool = nn.Sequential(
                            conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2,
                            layer3,
                            layer4,
                            nn.AvgPool2d(4),
                            nn.Conv2d(512, 32, kernel_size=1, padding=0, stride=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 50, kernel_size=1, padding=0, stride=1),
                            nn.ReLU(),
                            nn.Conv2d(50, 10, kernel_size=1, padding=0, stride=1),
                            ModuleLinearView(),
                            nn.Sequential(
                                nn.LeakyReLU(),
                                nn.Linear(10, dim_wideoutput),
                                nn.ReLU(),
                                nn.Linear(dim_wideoutput, num_classes)
                            )
                        )
        
        
        self.linear = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers_conv1bn1relu_1234_avgpool(x)
        return out


class ResNet18(ResNet):
    def __init__(self, dim_wideoutput, num_classes):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], dim_wideoutput, num_classes)


def test():
    net = ResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
