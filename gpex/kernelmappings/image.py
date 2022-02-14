
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import copy
#import relatedwork
#from relatedwork.utils.generativemodels import SqueezeNet10Encoder, ResidualEncoder


class SqueezeNet10Encoder(nn.Module):
    '''
    This class implements the convolutional part (i.e., without heads) of squeezenet.
    '''
    def __init__(self, pretrained):
        super(SqueezeNet10Encoder, self).__init__()
        #grab privates =========
        self.pretrained = pretrained
        #make conv_model ======
        model = torchvision.models.squeezenet1_0()
        list_modules = [list(model.children())[0]]
        #list_modules[-1][-1].relu = ModuleIdentity()
        self.model = nn.Sequential(*list_modules)
    
    def forward(self, x):
        return self.model(x)



class ResidualEncoder(nn.Module):
    '''
    This class implements the convolutional part (i.e., without heads) of resnet.
    '''
    def __init__(self, resnettype, pretrained):
        super(ResidualEncoder, self).__init__()
        #grab privates =========
        self.resnettype = resnettype
        self.pretrained = pretrained
        #make conv_model ======
        model = resnettype(pretrained = self.pretrained)
        list_modules = list(model.children())[0:-2]
        #list_modules[-1][-1].relu = ModuleIdentity()
        self.model = nn.Sequential(*list_modules)
    
    def forward(self, x):
        return self.model(x)


class AkNormalize(torch.nn.Module):
    def __init__(self):
        super(AkNormalize, self).__init__()
        
    def forward(self, x):
        return torch.nn.functional.normalize(x)*0.1 #---override TODO:check

class ImgtoVectorView(nn.Module):
    def __init__(self):
        super(ImgtoVectorView, self).__init__()
    
    def forward(self, x):
        assert(len(list(x.size())) == 4)
        return x[:,:,0,0]



class CNNList(nn.Module):
    '''
    This class implements a list of CNNs the output of which are all concatenated together.
    '''
    def __init__(self, num_classes, du_per_class, type_backbone, kwargs_backbone):
        super(CNNList, self).__init__()
        #grab args ===
        self.num_classes = num_classes
        self.du_per_class = du_per_class
        self.type_backbone = type_backbone
        self.kwargs_backbone = kwargs_backbone
        
        #infer the output channels of the backbone module ===
        x = torch.randn((2,3,224,224))
        temp_backbone = self.type_backbone(**self.kwargs_backbone)
        netout_temp = temp_backbone(x)
        C_backboneout = (netout_temp.size())[1]
        print("C_backboneout = {}".format(C_backboneout))
        
        list_beforeavgpool, list_afteravgpool = [], [] #separate the modules before (and inc.) the avg pooling and afterwards. 
        for c in range(num_classes):
            list_beforeavgpool.append(
                torch.nn.Sequential(
                    self.type_backbone(**self.kwargs_backbone)
                )
            )
            list_afteravgpool.append(
                        torch.nn.Sequential(
                            nn.AdaptiveAvgPool2d((1,1)),
                            ImgtoVectorView(),
                            nn.Linear(C_backboneout, self.du_per_class),
                            AkNormalize(),
                            nn.LeakyReLU()
                    )
            )
            
        self.list_beforeavgpool = nn.ModuleList(list_beforeavgpool)
        self.list_afteravgpool = nn.ModuleList(list_afteravgpool)    
        self._rng_outputheads = None
        
    def set_rng_outputheads(self, rng_outputhead):
        '''
        To be called by TGP when `flag_train_memefficient` is set to True.
        '''
        self._rng_outputheads = rng_outputhead
    
    def forward_untilbeforeavgpooling(self, x):
        list_output = []
        if(self._rng_outputheads is None):
            for n in range(self.num_classes):
                list_output.append(
                        self.list_beforeavgpool[n](
                            x
                    )
                )
        else:
            for n in range(self._rng_outputheads[1] - self._rng_outputheads[0]):
                list_output.append(
                        self.list_beforeavgpool[n+self._rng_outputheads[0]](
                            x
                        )
                )
                
        #output = torch.cat(list_output, 1)
        return list_output #[N x C x h x w]
        
    
    def forward(self, x):
        list_output = []
        if(self._rng_outputheads is None):
            for n in range(self.num_classes):
                list_output.append(
                    self.list_afteravgpool[n](
                        self.list_beforeavgpool[n](
                            x
                        )
                    )
                )
        else:
            for n in range(self._rng_outputheads[1] - self._rng_outputheads[0]):
                list_output.append(
                    self.list_afteravgpool[n+self._rng_outputheads[0]](
                        self.list_beforeavgpool[n+self._rng_outputheads[0]](
                            x
                        )
                    )
                )
                
        output = torch.cat(list_output, 1)
        return output #[N x C]



class Resnet18List(CNNList):
    def __init__(self, num_classes, du_per_class, scale_macrokernel):
        super(Resnet18List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block":BasicBlock,
                    "num_blocks":[2, 2, 2, 2],
                    "scale_macrokernel":scale_macrokernel
                 }
          )


class Resnet34List(CNNList):
    def __init__(self, num_classes, du_per_class, scale_macrokernel):
        super(Resnet34List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block":BasicBlock,
                    "num_blocks":[3, 4, 6, 3],
                    "scale_macrokernel":scale_macrokernel
                 }
          )

class Resnet50List(CNNList):
    def __init__(self, num_classes, du_per_class, scale_macrokernel):
        super(Resnet50List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block":Bottleneck,
                    "num_blocks":[3, 4, 6, 3],
                    "scale_macrokernel":scale_macrokernel
                 }
          )
          


class Resnet101List(CNNList):
    def __init__(self, num_classes, du_per_class, scale_macrokernel):
        super(Resnet101List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block":Bottleneck,
                    "num_blocks": [3, 4, 23, 3],
                    "scale_macrokernel":scale_macrokernel
                 }
          )
        

class Resnet152List(CNNList):
    def __init__(self, num_classes, du_per_class, scale_macrokernel):
        super(Resnet152List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": Bottleneck,
                    "num_blocks": [3, 8, 36, 3],
                    "scale_macrokernel":scale_macrokernel
                 }
          )

class SqueezeNetList(CNNList):
    def __init__(self, num_classes, du_per_class):
        super(SqueezeNetList, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = SqueezeNet10Encoder,
                kwargs_backbone = {
                    "pretrained":False
                 }
          )
          
          
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
          
class TinyResNetBackbone(nn.Module):
    def __init__(self, scale_macrokernel, block, num_blocks):
        super(TinyResNetBackbone, self).__init__()
        #grab args ===
        self.scale_macrokernel = scale_macrokernel
        
        
        #make the resnet module ===
        self.in_planes = int(64/scale_macrokernel)
        conv1 = nn.Conv2d(3, int(64/scale_macrokernel), kernel_size=3,
                        stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(int(64/scale_macrokernel))
        layer1 = self._make_layer(block, int(64/scale_macrokernel), num_blocks[0], stride=1)
        layer2 = self._make_layer(block, int(128/scale_macrokernel), num_blocks[1], stride=2)
        layer3 = self._make_layer(block, int(256/scale_macrokernel), num_blocks[2], stride=2)
        layer4 = self._make_layer(block, int(512/scale_macrokernel), num_blocks[3], stride=2)
        self.module = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            layer1,
            layer2,
            layer3,
            layer4,
        )
    
    def forward(self, x):
        return self.module(x)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)



class ResnetBackbone(nn.Module):
    def __init__(self, block, num_blocks, scale_macrokernel):
        super(ResnetBackbone, self).__init__()
        #grab args ===
        #scale_macrokernel = 1.0
        self.scale_macrokernel = scale_macrokernel
        
        
        #make the resnet module ===
        self.in_planes = int(64/scale_macrokernel)
        conv1 = nn.Conv2d(3, int(64/scale_macrokernel), kernel_size=3,
                        stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(int(64/scale_macrokernel))
        layer1 = self._make_layer(block, int(64/scale_macrokernel), num_blocks[0], stride=1)
        layer2 = self._make_layer(block, int(128/scale_macrokernel), num_blocks[1], stride=2)
        layer3 = self._make_layer(block, int(256/scale_macrokernel), num_blocks[2], stride=2)
        layer4 = self._make_layer(block, int(512/scale_macrokernel), num_blocks[3], stride=2)
        self.module = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            layer1,
            layer2,
            layer3,
            layer4
        )
        
        
    def forward(self, x):
        return self.module(x)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)




class ResnetBackboneKernel(nn.Module):
    def __init__(self, block, num_blocks, num_classes, du_per_class):
        super(ResnetBackboneKernel, self).__init__()
        #grab args ===
        scale_macrokernel = 1.0
        self.scale_macrokernel = 1.0
        
        
        #make the resnet module ===
        self.in_planes = int(64/scale_macrokernel)
        conv1 = nn.Conv2d(3, int(64/scale_macrokernel), kernel_size=3,
                        stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(int(64/scale_macrokernel))
        layer1 = self._make_layer(block, int(64/scale_macrokernel), num_blocks[0], stride=1)
        layer2 = self._make_layer(block, int(128/scale_macrokernel), num_blocks[1], stride=2)
        layer3 = self._make_layer(block, int(256/scale_macrokernel), num_blocks[2], stride=2)
        layer4 = self._make_layer(block, int(512/scale_macrokernel), num_blocks[3], stride=2)
        self.module1 = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            layer1,
            layer2,
            layer3,
            layer4
        )
        #infer the number of output heads ===
        with torch.no_grad():
            x = torch.randn(5,3,200,200)
            C_head = self.module1(x).size()[1]
            print("C_head is equal to {}.".format(C_head))
        self.module2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ImgtoVectorView(),
            nn.Linear(C_head, num_classes*du_per_class),
            AkSplitDimNormalize(num_classes = num_classes, dim_per_class = du_per_class),
            nn.LeakyReLU()
        )
        
        
    
    def forward(self, x):
        return self.module2(self.module1(x))
        
    def forward_untilbeforeavgpooling(self, x):
        return self.module1(x)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class Resnet18BackboneKernel(ResnetBackboneKernel):
    def __init__(self, num_classes, du_per_class):
        super(Resnet18BackboneKernel, self).__init__(
            block = BasicBlock,
            num_blocks = [2, 2, 2, 2],
            num_classes = num_classes,
            du_per_class = du_per_class
        )

class Resnet34BackboneKernel(ResnetBackboneKernel):
    def __init__(self, num_classes, du_per_class):
        super(Resnet34BackboneKernel, self).__init__(
            block = BasicBlock,
            num_blocks = [3, 4, 6, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )

class Resnet50BackboneKernel(ResnetBackboneKernel):
    def __init__(self, num_classes, du_per_class):
        super(Resnet50BackboneKernel, self).__init__(
            block = Bottleneck,
            num_blocks = [3, 4, 6, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )



class ResnetBackboneKernelDivideAfterAvgPool(nn.Module):
    def __init__(self, block, num_blocks, num_classes, du_per_class):
        super(ResnetBackboneKernelDivideAfterAvgPool, self).__init__()
        #grab args ===
        scale_macrokernel = 1.0
        self.scale_macrokernel = 1.0
        
        
        #make the resnet module ===
        self.in_planes = int(64/scale_macrokernel)
        conv1 = nn.Conv2d(3, int(64/scale_macrokernel), kernel_size=3,
                        stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(int(64/scale_macrokernel))
        layer1 = self._make_layer(block, int(64/scale_macrokernel), num_blocks[0], stride=1)
        layer2 = self._make_layer(block, int(128/scale_macrokernel), num_blocks[1], stride=2)
        layer3 = self._make_layer(block, int(256/scale_macrokernel), num_blocks[2], stride=2)
        layer4 = self._make_layer(block, int(512/scale_macrokernel), num_blocks[3], stride=2)
        self.module1 = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            layer1,
            layer2,
            layer3,
            layer4
        )
        #infer the number of output heads ===
        with torch.no_grad():
            x = torch.randn(5,3,200,200)
            C_head = self.module1(x).size()[1]
            print("C_head is equal to {}.".format(C_head))
        self.module2 = nn.Sequential(
            nn.Conv2d(C_head, 100*num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(100*num_classes, num_classes*du_per_class, kernel_size=1, stride=1, padding=0)
        )
        self.module3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ImgtoVectorView(),
            AkSplitDimNormalize(num_classes = num_classes, dim_per_class = du_per_class),
            nn.LeakyReLU()
        )
        
        
    
    def forward(self, x):
        return self.module3(self.module2(self.module1(x))).unsqueeze(-1).unsqueeze(-1)
    
        
    def forward_untilbeforeavgpooling(self, x):
        return self.module2(self.module1(x))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class Resnet18BackboneKernelDivideAvgPool(ResnetBackboneKernelDivideAfterAvgPool):
    def __init__(self, num_classes, du_per_class):
        super(Resnet18BackboneKernelDivideAvgPool, self).__init__(
            block = BasicBlock,
            num_blocks = [2, 2, 2, 2],
            num_classes = num_classes,
            du_per_class = du_per_class
        )



class Resnet34BackboneKernelDivideAvgPool(ResnetBackboneKernelDivideAfterAvgPool):
    def __init__(self, num_classes, du_per_class):
        super(Resnet34BackboneKernelDivideAvgPool, self).__init__(
            block = BasicBlock,
            num_blocks = [3, 4, 6, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )

class Resnet50BackboneKernelDivideAvgPool(ResnetBackboneKernelDivideAfterAvgPool):
    def __init__(self, num_classes, du_per_class):
        super(Resnet50BackboneKernelDivideAvgPool, self).__init__(
            block = Bottleneck,
            num_blocks = [3, 4, 6, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )
    
    
    
class Resnet101BackboneKernelDivideAvgPool(ResnetBackboneKernelDivideAfterAvgPool):
    def __init__(self, num_classes, du_per_class):
        super(Resnet101BackboneKernelDivideAvgPool, self).__init__(
            block = Bottleneck,
            num_blocks = [3, 4, 23, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )
        
class Resnet152BackboneKernelDivideAvgPool(ResnetBackboneKernelDivideAfterAvgPool):
    def __init__(self, num_classes, du_per_class):
        super(Resnet152BackboneKernelDivideAvgPool, self).__init__(
            block = Bottleneck,
            num_blocks = [3, 8, 36, 3],
            num_classes = num_classes,
            du_per_class = du_per_class
        )

class TinyResNet18List(CNNList):
    def __init__(self, scale_macrokernel, num_classes, du_per_class):
        super(TinyResNet18List, self).__init__(
                num_classes,
                du_per_class,
                type_backbone = TinyResNetBackbone,
                kwargs_backbone = {
                    "scale_macrokernel":scale_macrokernel,
                    "block": BasicBlock,
                    "num_blocks": [2, 2, 2, 2]
                 }
          )



class AkSplitDimNormalize(torch.nn.Module):
    def __init__(self, dim_per_class=20, num_classes=9):
        super(AkSplitDimNormalize, self).__init__()
        self.dim_per_class = dim_per_class
        #make a list of dimranges ===
        list_dimrange = []
        for c in range(num_classes):
            list_dimrange.append(
                [c*dim_per_class , (c+1)*dim_per_class]
            )
        self.list_dimrange = list_dimrange
        
    def forward(self, x):
        list_toret = [
            torch.nn.functional.normalize(x[:, dimrng[0]:dimrng[1]])*0.1
            for dimrng in self.list_dimrange
        ]
        toret = torch.cat(list_toret, 1)
        return toret







class MultiBNListAndOneLayer(nn.Module):
    '''
    This class implements a list of CNNs the output of which are all concatenated together.
    '''
    def __init__(self, num_classes, du_per_class, num_backbones, type_backbone, kwargs_backbone):
        super(MultiBNListAndOneLayer, self).__init__()
        #grab args ===
        self.num_classes = num_classes
        self.du_per_class = du_per_class
        self.num_backbones = num_backbones
        self.type_backbone = type_backbone 
        self.kwargs_backbone = kwargs_backbone
        
        #infer the output channels of the backbone module ===
        x = torch.randn((2,3,224,224))
        temp_backbone = self.type_backbone(**self.kwargs_backbone)
        netout_temp = temp_backbone(x)
        C_backboneout = (netout_temp.size())[1]
        print("C_backboneout = {}".format(C_backboneout))
        
        #make list of backbones ===
        list_backbone = []
        for c in range(self.num_backbones):
            list_backbone.append(
                torch.nn.Sequential(
                    self.type_backbone(**self.kwargs_backbone)
                )
            )
        self.list_backbone = nn.ModuleList(list_backbone)
        
        #make afterbackbone modules ===
        self.module_afterbackbone_part1 = torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels = C_backboneout*num_backbones,
                        out_channels = num_classes * du_per_class,
                        kernel_size = 1,
                        stride=1,
                        padding=0
                    )
                )
        self.module_afterbackbone_part2 = torch.nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    ImgtoVectorView(),
                    AkSplitDimNormalize(
                        dim_per_class=du_per_class,
                        num_classes= num_classes
                    ),
                    nn.LeakyReLU()
                )
        self._rng_outputheads = None
        
    def set_rng_outputheads(self, rng_outputhead):
        '''
                To be called by TGP when `flag_train_memefficient` is set to True.
              '''
        self._rng_outputheads = rng_outputhead
    
    def forward(self, x):
        #pass x to backbones ===
        list_output = []
        N = x.size()[0]
        try:
            x_device = x.get_device() 
        except:
            x_device = None
        for n in range(self.num_backbones):
            if(self._rng_outputheads is not None):
                if(self._rng_outputheads[0] <= n < self._rng_outputheads[1]):
                    output_head = self.list_backbone[n](x)
                else:
                    output_head = torch.zeros((1, self.du_per_class)).float()
                    if(x_device is not None):
                        output_head = output_head.to(x_device).detach()
                    else:
                        output_head = output_head.detach()
                list_output.append(output_head)
            else:
                output_head = self.list_backbone[n](x)
                list_output.append(
                    output_head
                )
        output_backbone = torch.cat(list_output , 1) #[N x C_backboneout*num_backbones x h x w]
        
        #pass to the layer after backbones ===
        output = self.module_afterbackbone_part2(
                        self.module_afterbackbone_part1(
                            output_backbone
                    )
        )
        return output #[N x C]




class MultiResnet18ListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiResnet18ListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": BasicBlock,
                    "num_blocks": [2, 2, 2, 2]
                 }
          )


class MultiResnet34ListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiResnet34ListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": BasicBlock,
                    "num_blocks": [3, 4, 6, 3]
                 }
          )


class MultiResnet50ListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiResnet50ListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": Bottleneck,
                    "num_blocks": [3, 4, 6, 3]
                 }
          )


class MultiResnet101ListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiResnet101ListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": Bottleneck,
                    "num_blocks": [3, 4, 23, 3]
                 }
          )


class MultiResnet152ListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiResnet101ListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = ResnetBackbone,
                kwargs_backbone = {
                    "block": Bottleneck,
                    "num_blocks": [3, 8, 36, 3]
                 }
          )


          
class MultiSqueezeNetListAndOneLayer(MultiBNListAndOneLayer):
    def __init__(self, num_classes, du_per_class, num_backbones):
        super(MultiSqueezeNetListAndOneLayer, self).__init__(
                num_classes,
                du_per_class,
                num_backbones,
                type_backbone = SqueezeNet10Encoder,
                kwargs_backbone = {
                    "pretrained":False
                 }
          )


