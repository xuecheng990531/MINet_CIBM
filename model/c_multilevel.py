import torch.nn.functional as F
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

    
class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class aaca_layer(nn.Module):
    """Constructs a AACA module.

    Args:
        channel: Number of channels of the input feature map
        gamma, q: two constants, which indicate the slope and the y-intercept of the linear function
                  in a mapping between the channel dimension and kernel size
        lamda, m: two constants, which indicate the slope and the y-intercept of the linear function
                  in a mapping between the dilated rate and kernel size
    """

    def __init__(self, channel,gamma=2, q=1, lamda=2, m=1):
        super(aaca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + q) / gamma))
        ## k: kernel size  ##
        k = t if t % 2 else t - 1
        # r: dilated rate ##
        r = int((k + m) / lamda)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, dilation=r, padding=int((k - 1) * r / 2), bias=False)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=k, dilation=r, padding=int((k - 1) * r / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        ## feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)      
        y = y_avg

        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.conv_1(self.relu(y))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()

        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        return x


class BN_RELU(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


## adaptive atrous channel attention model ###
class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.aaca = aaca_layer(channel)

    def forward(self, x):
        x_aaca = self.aaca(x)
        out = x + x_aaca
        return out


class Attention_block(nn.Module):
    def __init__(self, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int // 2)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_int, F_int // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int // 2)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return psi


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4), bottleneck_layers=4,
                 growth_rate=12, out_chans_first_conv=48, n_classes=2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        self.channelAttDownBlocks = nn.ModuleList([])
        self.bn_relu = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            self.channelAttDownBlocks.append(ChannelAttention(cur_channels_count))
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.bn_relu.append(BN_RELU(cur_channels_count))
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count = prev_block_channels
        # print(cur_channels_count)
        self.channelAttBottleneck = ChannelAttention(cur_channels_count)
        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.channelAttUpBlocks = nn.ModuleList([])
        cur_channels = []
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(cur_channels_count, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=False))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
            cur_channels.append(cur_channels_count)
            self.channelAttUpBlocks.append(ChannelAttention(cur_channels_count))

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            cur_channels_count, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        cur_channels.append(cur_channels_count)
        # print(cur_channels_count)
        self.channelAttUpBlocks.append(ChannelAttention(cur_channels_count))

        ## multi-level attention model ##
        self.down4 = nn.Sequential(
            nn.Conv2d(cur_channels[0], 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(cur_channels[1], 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(cur_channels[2], 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(cur_channels[3], 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.attention4 = Attention_block(64)
        self.attention3 = Attention_block(64)
        self.attention2 = Attention_block(64)
        self.attention1 = Attention_block(64)


        self.refine4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.predict4 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict3 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict1 = nn.Conv2d(64, n_classes, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, n_classes, kernel_size=1)

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            ## aaca module ##
            out = self.channelAttDownBlocks[i](out)
            out1 = self.bn_relu[i](out)
            skip_connections.append(out1)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        out = self.channelAttBottleneck(out)
        skip = skip_connections.pop()
        out = self.transUpBlocks[0](out, skip)
        layer4 = self.denseBlocksUp[0](out)
        skip = skip_connections.pop()
        out = self.transUpBlocks[1](layer4, skip)
        layer3 = self.denseBlocksUp[1](out)
        skip = skip_connections.pop()
        out = self.transUpBlocks[2](layer3, skip)
        layer2 = self.denseBlocksUp[2](out)
        skip = skip_connections.pop()
        out = self.transUpBlocks[3](layer2, skip)
        layer1 = self.denseBlocksUp[3](out)

        ## multi-level attention ##
        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(down4, fuse1)
        attention3 = self.attention3(down3, fuse1)
        attention2 = self.attention2(down2, fuse1)
        attention1 = self.attention1(down1, fuse1)

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)
        final = (predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4

        return self.sigmoid(final)

def Net (in_channels, n_classes):
    return FCDenseNet(
        in_channels=in_channels, down_blocks=(4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)

if __name__=='__main__':
    x=torch.randn(1,1,128,128)
    model=FCDenseNet(in_channels=1, down_blocks=(4, 4, 4, 4),up_blocks=(4, 4, 4, 4), bottleneck_layers=4,growth_rate=12, out_chans_first_conv=48, n_classes=1)
    print(model(x).shape)
