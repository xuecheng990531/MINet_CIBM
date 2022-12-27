from multiprocessing import pool
from unittest import result
import torch
import sys
import numpy as np
sys.path.append('utils')
import torch.nn.functional as F
from torch import nn
from aspp import *

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

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

def concat_conv(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)
        

class MINet(nn.Module):
    def __init__(self, class_number=5, in_channels=3):
        super().__init__()
        if in_channels==1:
            self.conv2_1 = conv3x3_bn_relu(65, 128)
            self.conv3_1 = conv3x3_bn_relu(129, 256)
            self.conv4_1 = conv3x3_bn_relu(256, 512)
        elif in_channels==2:
            self.conv2_1 = conv3x3_bn_relu(66, 128)
            self.conv3_1 = conv3x3_bn_relu(130, 256)
            self.conv4_1 = conv3x3_bn_relu(256, 512)

        self.cbam1=CBAM(channel=64)
        self.cbam2=CBAM(channel=128)
        self.cbam3=CBAM(channel=256)
        self.cbam4=CBAM(channel=512)
        # encoder
        self.conv1_1 = conv3x3_bn_relu(in_channels, 64)
        self.conv1_2 = conv3x3_bn_relu(64, 64)

        self.maxpool = nn.MaxPool2d(2, 2)  # only one for all

        self.conv2_2 = conv3x3_bn_relu(128, 128)
        self.conv3_2 = conv3x3_bn_relu(256, 256)
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

        self.conv1_aspp=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.conv2_aspp=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )


        self.conv3_aspp=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.conv4_aspp=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.reduce=nn.Sequential(
            nn.Conv2d(in_channels=1024*5,out_channels=1024*2,kernel_size=1),
            nn.BatchNorm2d(1024*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024*2,out_channels=1024,kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.aspp=ASPP(in_channel=1024,depth=1024)

    def forward(self, x):
        # 多个尺度的照片
        x_256=self.maxpool(x)
        x_128=self.maxpool(x_256)
        x_64=self.maxpool(x_128)

        # encoder
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        pool1 = self.cbam1(self.maxpool(conv1_2))
        

        cat_256=concat_conv(pool1,x_256)
        conv2_1 = self.conv2_1(cat_256)
        conv2_2 = self.conv2_2(conv2_1)

        pool2 = self.cbam2(self.maxpool(conv2_2))

        cat_128=concat_conv(pool2,x_128)
        conv3_1 = self.conv3_1(cat_128)
        conv3_2 = self.conv3_2(conv3_1)

        pool3 = self.cbam3(self.maxpool(conv3_2))

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)

        pool4 = self.cbam4(self.maxpool(conv4_2))

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        # decoder
        up6 = upsample(conv5_2, conv4_2)
        
        conv6 = self.conv6(up6)
        
        merge6 = concat_conv(conv6, conv4_2)
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        up7 = upsample(conv6_2, conv3_2)
        conv7 = self.conv7(up7)
        merge7 = concat_conv(conv7, conv3_2)
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        up8 = upsample(conv7_2, conv2_2)
        conv8 = self.conv8(up8)
        merge8 = concat_conv(conv8, conv2_2)
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        up9 = upsample(conv8_2, conv1_2)
        conv9 = self.conv9(up9)
        merge9 = concat_conv(conv9, conv1_2)
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)

        score = self.sigmoid(self.score(conv9_2))

        return score


if __name__=='__main__':
    model=MINet(class_number=1,in_channels=1).cuda() 
    print('model output shape:',model(torch.randn(1,1,512,512).cuda()).shape)