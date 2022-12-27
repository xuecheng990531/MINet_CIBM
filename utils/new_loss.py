import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return focal_loss

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
            IoU1 = Iand1/(Ior1 + 1e-5)
            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        return IoU/b

class Focal_IoU(nn.Module):
    def __init__(self,theta) -> None:
        super().__init__()
        self.focal=FocalLoss()
        self.iou=IoU_loss()
        self.theta=theta
    def forward(self,pred,target):
        return self.theta*self.focal(pred,target)+(1-self.theta)*self.iou(pred,target)

if __name__=='__main__':
    import torch
    x=torch.ones(1,1,512,512)
    y=torch.zeros(1,1,512,512)
    loss=FocalLoss()
    loss2=IoU_loss()
    loss3=Focal_IoU()
    print(loss2(x,y))
    print(loss3(x,y))
    print(loss(x,y))