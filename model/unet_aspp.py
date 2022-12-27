""" Parts of the U-Net model """
from aspp import  ASPP
import  torch
from torch import  nn
import  torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()  # 初始化权重
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.aspp = ASPP(in_channel=512 * 4, depth=512 * 4)
        self.out=nn.Sequential(
            nn.Conv2d(512*4,512*2,3,1),
            nn.ReLU(),
            nn.Conv2d(512*2,512,3,1),
            nn.ReLU()
        )
        self.conv64=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=512,kernel_size=3,stride=16),
            nn.ReLU()
        )
        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=8),
            nn.ReLU()
        )
        self.conv256 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=4),
            nn.ReLU()
        )

        self.sigmoid=nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x_256=self.maxpool(x)#torch.Size([1, 3, 256, 256])
        x_128=self.maxpool(x_256)#torch.Size([1, 3, 128, 128])
        x_64=self.maxpool(x_128)#torch.Size([1, 3, 64, 64])

        x1 = self.inc(x)
        x1_conv64=self.conv64(x1)

        x2 = self.down1(x1)
        x2_conv128=self.conv128(x2)

        x3 = self.down2(x2)
        x3_conv256 = self.conv256(x3)
        
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ffusion=torch.cat([x1_conv64,x2_conv128,x3_conv256,x5],dim=1)
        ffusion_result=self.aspp(ffusion)
        x5=self.out(ffusion_result)

        xa = self.up1(x5, x4)
        xb = self.up2(xa, x3)
        xc = self.up3(xb, x2)
        xd = self.up4(xc, x1)
        


        x = self.outc(xd)
        return self.sigmoid(x)


if __name__=='__main__':
    x=torch.randn(1,1,512,512)
    model=UNet(n_channels=1,n_classes=1)
    print(model(x).shape)