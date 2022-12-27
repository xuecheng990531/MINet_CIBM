from __future__ import division
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

# in_channel是输入特征通道维度
# depth是输出特征通道维度
class ASPP_Block(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP_Block,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.relu=nn.ReLU()
 
    def forward(self, x):
        size = x.shape[2:]
 		# mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # upsample.shape = torch.Size([8, 3, 32, 32])
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        
 		
 		# block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.relu(self.atrous_block1(x))

 		
 		# block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.relu(self.atrous_block6(x))
 		
 		# block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.relu(self.atrous_block12(x))
 		
 		# block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.relu(self.atrous_block18(x))
 		
 		# torch.cat.shape = torch.Size([8, 15, 32, 32])
 		# conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1))
        return net



if __name__=='__main__':
    x=torch.randn(1,3,512,512)
    model=ASPP_Block(in_channel=3,depth=64)
    print(model(x).shape)