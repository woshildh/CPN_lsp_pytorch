import torch.nn as nn
import torch.nn.functional as F
from .bottleneck import Bottleneck
import math
import torch
__all__=['CPNResNet50']

class GlobalNet(nn.Module):
    def __init__(self, block, layers, num_kps=14):
        self.inplanes = 64
        self.num_kps = num_kps
        super(GlobalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # make lateral
        self.make_lateral()
        # make top layer
        self.make_toplayer()
        # init weights
        self.__initweights__()
    def make_lateral(self):
        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
    def make_toplayer(self):
        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, self.num_kps, kernel_size=3, stride=1, padding=1)

    def __initweights__(self):   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
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
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self,x,y):
        _,_,h,w = y.size()
        return F.interpolate(x,(h,w),mode='bilinear') + y
    
    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        # get feature map
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # pyramid
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5,self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4,self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3,self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2,p3,p4,p5

class RefineNet(nn.Module):
    def __init__(self, num_kps=14):
        super(RefineNet, self).__init__()
        downsample = nn.Sequential(
                nn.Conv2d(num_kps, 256,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
        )
        self.bottleneck2 = Bottleneck(num_kps, 64, 1, downsample)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        downsample = nn.Sequential(
                nn.Conv2d(1024, 256,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
            )
        self.output = nn.Sequential(Bottleneck(1024, 64, 1,downsample),
                                    nn.Conv2d(256, num_kps, kernel_size=1, stride=1, padding=0))
        self.__initweights__()
    def __initweights__(self):   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))

class CPNResNet50(nn.Module):
    def __init__(self,num_kps=14,pretrained=None):
        super(CPNResNet50,self).__init__()
        self.num_kps = num_kps
        self.globalnet = GlobalNet(Bottleneck,[3,4,6,3],num_kps)
        self.refinenet = RefineNet(num_kps)
    def forward(self,x):
        p2,p3,p4,p5 = self.globalnet(x)
        out = self.refinenet(p2,p3,p4,p5)
        return out,p2

def cpn_resnet50(num_kps=14,pretrained=None):
    model = CPNResNet50(num_kps)
    if pretrained:
        model.globalnet.load_state_dict(torch.load(pretrained),strict=False)
        print("pretrained model load succeed")
    return model