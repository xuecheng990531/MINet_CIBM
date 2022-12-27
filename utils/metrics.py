#针对多分类问题，二分类问题更简单一点。
import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftIoULoss(nn.Module):
    def __init__(self, n_classes,l):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.bce=nn.BCELoss()
        self.lamda=l

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        # target_onehot = self.to_one_hot(target, self.n_classes)
        target_onehot=target

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -self.lamda*loss.mean()+self.bce(input,target)


if __name__=='__main__':
    import torch
    x=torch.zeros(1,1,512,512)
    y=torch.ones(1,1,512,512)

    loss=SoftIoULoss(n_classes=1,l=0.5)
    loss2=nn.BCELoss()


    print(loss(x,y))
    print(loss2(x,y))