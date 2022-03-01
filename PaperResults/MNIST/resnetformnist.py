'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgtoVectorView(nn.Module):
    def __init__(self):
        super(ImgtoVectorView, self).__init__()
    
    def forward(self, x):
        assert(len(list(x.size())) == 4)
        return x[:,:,0,0]



class Clamp(torch.nn.Module):
    def __init_(self, minval=-1.0, maxval=1.0):
        self.minval, self.maxval = minval, maxval
        super(Clamp, self).__init__()
        
        
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
    def __init__(self, block, num_blocks, num_classes=10, dim_wideoutput=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dim_wideoutput = dim_wideoutput
        self.block = block
        self.num_blocks = num_blocks
        conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        print(">>>>>>> self.dim_wideoutput = {}".format(self.dim_wideoutput))
        
        
        
        # ~ self.module_debug = nn.Sequential(
                            # ~ conv1,
                            # ~ bn1,
                            # ~ nn.ReLU(),
                            # ~ layer1,
                            # ~ layer2,
                            # ~ layer3,
                            # ~ layer4
                        # ~ )
        
        
        
        self.layers_conv1bn1relu_1234_avgpool = nn.Sequential(
                            conv1,
                            bn1,
                            nn.ReLU(),
                            layer1,
                            layer2,
                            layer3,
                            layer4,
                            nn.AdaptiveAvgPool2d((1,1)),
                            nn.Conv2d(512, 32, kernel_size=1, padding=0, stride=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 50, kernel_size=1, padding=0, stride=1),
                            nn.ReLU(),
                            nn.Conv2d(50, 10, kernel_size=1, padding=0, stride=1),
                            ModuleLinearView(),
                            nn.Sequential(
                                nn.LeakyReLU(),
                                nn.Linear(10, self.dim_wideoutput),
                                nn.ReLU(),
                                nn.Linear(self.dim_wideoutput, num_classes)
                            )
                        )
        
        
        self.linear = None #nn.Sequential(
                            # ~ nn.LeakyReLU(),
                            # ~ nn.Linear(10, 16),
                            # ~ nn.ReLU(),
                            # ~ nn.Linear(16, num_classes)
                         # ~ )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # ~ out = F.relu(self.bn1(self.conv1(x)))
        # ~ out = self.layer1(out)
        # ~ out = self.layer2(out)
        # ~ out = self.layer3(out)
        # ~ out = self.layer4(out)
        out = self.layers_conv1bn1relu_1234_avgpool(x)
        #print("out.shape = {}".format(out.shape)) #[N x 512 x 1 x 1]
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out


class ResNet18(ResNet):
    def __init__(self, num_classes=10, dim_wideoutput=10):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes, dim_wideoutput)


# ~ def AkResNet18():
    # ~ return AkResNet(BasicBlock, [2, 2, 2, 2])



def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = AkResNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()



class ResNetBackbone(nn.Module):
    def __init__(self, block, num_blocks, dim_before_wideoutput=None, input_size=32, dim_wideoutput=None, flag_setdimoutput_to_one = False):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64
        self.block = block
        self.dim_before_wideoutput = dim_before_wideoutput
        self.dim_wideoutput = dim_wideoutput
        self.num_blocks = num_blocks
        conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if(dim_before_wideoutput is not None):
            assert(dim_wideoutput is not None)
            if(flag_setdimoutput_to_one == True):
                self.layers_conv1bn1relu_1234 = nn.Sequential(
                                    conv1,
                                    bn1,
                                    nn.ReLU(),
                                    layer1,
                                    layer2,
                                    layer3,
                                    layer4,
                                    nn.Conv2d(512, self.dim_before_wideoutput, kernel_size=1, padding=0, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(self.dim_before_wideoutput, self.dim_wideoutput, kernel_size=1, padding=0, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(self.dim_wideoutput, 1, kernel_size=1, padding=0, stride=1),
                                    torch.nn.BatchNorm2d(1)
                                )
            else:
                self.layers_conv1bn1relu_1234 = nn.Sequential(
                                    conv1,
                                    bn1,
                                    nn.ReLU(),
                                    layer1,
                                    layer2,
                                    layer3,
                                    layer4,
                                    nn.Conv2d(512, self.dim_before_wideoutput, kernel_size=1, padding=0, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(self.dim_before_wideoutput, self.dim_wideoutput, kernel_size=1, padding=0, stride=1),
                                )
        else:
            if(flag_setdimoutput_to_one == True):
                self.layers_conv1bn1relu_1234 = nn.Sequential(
                                    conv1,
                                    bn1,
                                    nn.ReLU(),
                                    layer1,
                                    layer2,
                                    layer3,
                                    layer4,
                                    nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1),
                                    torch.nn.BatchNorm2d(1)
                                )
            else:
                self.layers_conv1bn1relu_1234 = nn.Sequential(
                                    conv1,
                                    bn1,
                                    nn.ReLU(),
                                    layer1,
                                    layer2,
                                    layer3,
                                    layer4
                                )
        #set the output size ====
        with torch.no_grad():
            netout = self.layers_conv1bn1relu_1234(
                torch.randn((5, 3, input_size, input_size))
            )
            self.size_output = list(netout.size())[1::]
            print("self.size_output was set to {}.".format(self.size_output))
        
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        

    def forward(self, x):
        '''
        Returns a tensor of shape [N x C*h*w x 1 x 1]
        '''
        out = self.layers_conv1bn1relu_1234(x) #[N x C x h x w].
        out = torch.reshape(out , [out.size()[0], out.size()[1]*out.size()[2]*out.size()[3], 1, 1])
        return out


class ResnetClassifierWithAttention(nn.Module):
    def __init__(
                self, num_classes, block_classifier, num_blocks_classifier,
                block_attention, num_blocks_attention, dim_before_wideoutput_attention,
                dim_wideoutput_attention, input_size
        ):
        super(ResnetClassifierWithAttention, self).__init__()
        
        #make the two internal modules ====
        self.backbone_classification = ResNetBackbone(
            block = block_classifier,
            num_blocks = num_blocks_classifier,
            dim_before_wideoutput=None,
            dim_wideoutput=None,
            input_size =input_size
        )
        self.module_attention = ResNetBackbone(
            block = block_attention,
            num_blocks = num_blocks_attention,
            dim_before_wideoutput = dim_before_wideoutput_attention,
            dim_wideoutput = dim_wideoutput_attention,
            flag_setdimoutput_to_one = True,
            input_size = input_size
        )
        self.module_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ModuleLinearView(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, y, n):
        out_classific = self.backbone_classification(x) #[N x C*w*h x 1 x 1]
        out_attention = self.module_attention(x) #[N x 1*w*h x 1 x 1]
        
        out_classific = torch.reshape(
                    out_classific[:, :, 0, 0],
                    [x.size()[0], self.backbone_classification.size_output[0], self.backbone_classification.size_output[1], self.backbone_classification.size_output[2]]
        )  #[N x C x h x w]
        #print("x.shape = {}".format(x.shape))
        #print("out_attention.shape =  {}".format(out_attention.shape))
        out_attention = torch.reshape(
                out_attention[:, :, 0, 0], 
                [x.size()[0], self.module_attention.size_output[0], self.module_attention.size_output[1], self.module_attention.size_output[2]]
        ) #[N x 1 x h x w]
        out_attention = torch.nn.functional.sigmoid(out_attention) #[N x 1 x h x w] in (0.0, 1.0)
        min_attention_clamp, max_attention_clamp = 0.1, 0.9
        out_attention = torch.clamp(out_attention, min=min_attention_clamp, max=max_attention_clamp)
        
        normalization_attention = torch.sum(out_attention, 3) #[N x 1 x h]
        normalization_attention = torch.sum(normalization_attention, 2) #[N x 1]
        normalization_attention = normalization_attention.unsqueeze(-1).unsqueeze(-1) #[N x 1 x 1 x 1]
        toret = self.module_linear((out_classific * out_attention)/normalization_attention) #[N x num_classes]
        return toret, y, n
        
