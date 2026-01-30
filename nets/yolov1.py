import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

class CBR(nn.Module):
    def __init__(self, in_c, out_c, ksize, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, ksize, stride, pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.LReLU = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LReLU(x)
        return x

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, B=2, ic_debug=False):
        super().__init__()
        self.ic_debug = ic_debug
        self.backbone = nn.Sequential(
            CBR(3, 64, 7, 2, 3), # conv1
            nn.MaxPool2d(2, 2), # pool1 ([2, 64, 112, 112])
            CBR(64, 192, 3, 1, 1), # conv2
            nn.MaxPool2d(2, 2), # pool2 ([2, 192, 56, 56])
            CBR(192, 128, 1, 1, 0), # conv3
            CBR(128, 256, 3, 1, 1), # conv4
            CBR(256, 256, 1, 1, 0), # conv5
            CBR(256, 512, 3, 1, 1), # conv6
            nn.MaxPool2d(2, 2), # pool3 ([2, 512, 28, 28])
            CBR(512, 256, 1, 1, 0), # conv7
            CBR(256, 512, 3, 1, 1), # conv8
            CBR(512, 256, 1, 1, 0), # conv9
            CBR(256, 512, 3, 1, 1), # conv10
            CBR(512, 256, 1, 1, 0), # conv11
            CBR(256, 512, 3, 1, 1), # conv12
            CBR(512, 256, 1, 1, 0), # conv13
            CBR(256, 512, 3, 1, 1), # conv14
            CBR(512, 512, 1, 1, 0), # conv15
            CBR(512, 1024, 3, 1, 1), # conv16
            nn.MaxPool2d(2, 2), # pool4 ([2, 1024, 14, 14])
            CBR(1024, 512, 1, 1, 0), # conv17
            CBR(512, 1024, 3, 1, 1), # conv18
            CBR(1024, 512, 1, 1, 0), # conv19
            CBR(512, 1024, 3, 1, 1), # conv20
        )
        self.head_conv = nn.Sequential(
            CBR(1024, 1024, 3, 1, 1), # conv21 ([2, 1024, 14, 14])
            CBR(1024, 1024, 3, 2, 1), # conv22 ([2, 1024, 7, 7])
            CBR(1024, 1024, 3, 1, 1), # conv23 ([2, 1024, 7, 7])
            CBR(1024, 1024, 3, 1, 1), # conv24 ([2, 1024, 7, 7])
        )
        
        self.fc1 = nn.Linear(1024*7*7, 4096)
        self.LReLU = nn.LeakyReLU(0.1, inplace=True)
        # self.ReLU = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 7*7*(num_classes+B*5))
        
    def forward(self, x):
        x_backbone = self.backbone(x)
        x_head_conv = self.head_conv(x_backbone)
        x = x_head_conv.view(x_head_conv.shape[0], -1)
        x = self.fc1(x)
        x = self.LReLU(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 7, 7, -1)
        
        if self.ic_debug:
            print('head_conv output shape:')
            ic(x_head_conv.shape)
        if self.ic_debug:
            print('backbone output shape:')
            ic(x_backbone.shape)
            
        if self.ic_debug:
            print('Object finnal output shape:')
            ic(x.shape)
        return x
    
class YOLOv1_Classifier(nn.Module):
    def __init__(self, num_classes=20, B=2, ic_debug=False):
        super().__init__()
        self.ic_debug = ic_debug
        backbone = YOLOv1(ic_debug=ic_debug)
        self.conv_backbone = backbone.backbone
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(1024, num_classes)
        
        
    def forward(self, x):
        x_backbone = self.conv_backbone(x)
        if self.ic_debug:
            print('backbone output shape:')
            ic(x_backbone.shape)
        x = self.gap(x_backbone)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.ic_debug:
            print('Classifier finnal output shape:')
            ic(x.shape)
        return x
if __name__ == "__main__":
    # classifier
    mode = 'classifier'
    
    if mode == 'train':
        model = YOLOv1(ic_debug=True)
        x = torch.randn(8, 3, 448, 448)
        y = model(x)
    
    if mode == 'classifier':
        model = YOLOv1_Classifier(num_classes=1000, ic_debug=True)
        x = torch.randn(8, 3, 224, 224)
        y = model(x)