import torch
import sys
sys.path.append('utils/')
import torch.nn.functional as F
from torch import nn
from CBAM import *

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))

def upsample(in_features, out_features):
    shape = out_features.shape[2:]  # h w
    return F.interpolate(in_features, size=shape, mode='bilinear', align_corners=True)

def concat(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)

class fusion(nn.Module):
    def __init__(self,chann) -> None:
        super(fusion,self).__init__()
        self.chann=chann
    def forward(self,encoder_block,upsample_block):
        subtract=(upsample_block-encoder_block)
        fusion=torch.cat([subtract,upsample_block],dim=1)
        
        return fusion
        

class U_Net(nn.Module):
    def __init__(self, class_number=5, in_channels=3):
        super().__init__()
        # fusion
        self.merge6=fusion(chann=1024)
        self.merge7=fusion(chann=512)
        self.merge8=fusion(chann=256)
        self.merge9=fusion(chann=128)


        # encoder
        self.conv1_1 = conv3x3_bn_relu(in_channels, 64)
        self.conv1_2 = conv3x3_bn_relu(64, 64)

        self.maxpool = nn.MaxPool2d(2, 2)  # only one for all

        self.conv2_1 = conv3x3_bn_relu(64, 128)
        self.conv2_2 = conv3x3_bn_relu(128, 128)

        self.conv3_1 = conv3x3_bn_relu(128, 256)
        self.conv3_2 = conv3x3_bn_relu(256, 256)

        self.conv4_1 = conv3x3_bn_relu(256, 512)
        self.conv4_2 = conv3x3_bn_relu(512, 512)

        self.conv5_1 = conv3x3_bn_relu(512, 1024)
        self.conv5_2 = conv3x3_bn_relu(1024, 1024)

        # decoder
        self.conv6 = conv3x3_bn_relu(1024, 512)
        self.conv6_1 = conv3x3_bn_relu(1024, 512)  ##
        self.conv6_2 = conv3x3_bn_relu(512, 512)

        self.conv7 = conv3x3_bn_relu(512, 256)
        self.conv7_1 = conv3x3_bn_relu(512, 256)  ##
        self.conv7_2 = conv3x3_bn_relu(256, 256)

        self.conv8 = conv3x3_bn_relu(256, 128)
        self.conv8_1 = conv3x3_bn_relu(256, 128)  ##
        self.conv8_2 = conv3x3_bn_relu(128, 128)

        self.conv9 = conv3x3_bn_relu(128, 64)
        self.conv9_1 = conv3x3_bn_relu(128, 64)  ##
        self.conv9_2 = conv3x3_bn_relu(64, 64)

        self.score = nn.Conv2d(64, class_number, 1, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        # encoder
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.maxpool(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.maxpool(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        # decoder
        up6 = upsample(conv5_2, conv4_2)
        
        conv6 = self.conv6(up6)
        
        # merge6 = concat(conv6, conv4_2)
        # 首先将上采样过的特征图和encoder的特征图做减法，得到信息丢失的区域得到新的特征residual_feature ，与上采样的特征图进行cat，得到的结果经过注意力机制进行处理
        merge6=self.merge6(conv6,conv4_2)
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        up7 = upsample(conv6_2, conv3_2)
        conv7 = self.conv7(up7)
        # merge7 = concat(conv7, conv3_2)
        merge7=self.merge7(conv7,conv3_2)
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        up8 = upsample(conv7_2, conv2_2)
        conv8 = self.conv8(up8)
        # merge8 = concat(conv8, conv2_2)
        merge8=self.merge8(conv8, conv2_2)
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        up9 = upsample(conv8_2, conv1_2)
        conv9 = self.conv9(up9)
        # merge9 = concat(conv9, conv1_2)
        merge9=self.merge9(conv9, conv1_2)
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)

        score = self.sigmoid(self.score(conv9_2))

        return score


if __name__=='__main__':
    model=U_Net(class_number=1,in_channels=3)
    print('model output shape:',model(torch.randn(1,3,512,512)).shape)