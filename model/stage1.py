import torch
from torch import nn
import torch.nn.functional as F

class stage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.con9=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=128,kernel_size=9,padding='same'),
            nn.ReLU()
        )
        self.con7=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=7,padding='same'),
            nn.ReLU()
        )
        self.down=nn.MaxPool2d(2, 2)
        self.con3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU()
        )
        self.con3_final=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU()
        )
        self.con1=nn.Sequential(
            nn.Conv2d(256,128,1,1),
            nn.ReLU(),
            nn.Conv2d(128,64,1,1),
            nn.ReLU(),
            nn.Conv2d(64,32,1,1),
            nn.ReLU(),
            nn.Conv2d(32,1,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        conv9=self.con9(x)
        conv7=self.con7(conv9)
        down=self.down(conv7)
        conv3=self.con3(down)
        up=F.interpolate(input=conv3,scale_factor=2)
        conv3_final=self.con3_final(up)

        return self.con1(conv3_final)


class refinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downconv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            nn.ReLU(),
        )
        self.down=nn.MaxPool2d(2,2)
        self.donw2conv=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU()
        )
        self.upconv=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding='same'),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        conv1=self.downconv(x)
        down=self.down(conv1)
        conv2=self.donw2conv(down)
        up=F.interpolate(conv2,scale_factor=2)
        conv3=self.upconv(up)

        return (conv3)


if __name__=='__main__':
    x=torch.randn(1,1,512,512)
    # y=torch.randn(1,1,512,512)
    model=refinement()
    print(model(x).shape)