
import os
import argparse
import torch

# from utils.data import HRFDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import sys
sys.path.append('..')
from utils.data_loader import fundus_data
from model.stage1 import stage1
from utils.new_loss import Focal_IoU

# 这是消融实验的模型
# from model.mymodel_MIF_SCIF import U_Net as mymodel
# from model.fcn import FCN
# from model.model_SCIF import U_Net as model_scif
# from model.model_MIF import U_Net as model_mif
# from model.baseline import U_Net as baseline
# from model.MIF import MINet as mif
# from model.MINet import MINet
# from model.MASP import MINet as masp
# from model.model_SCIF_noattention import U_Net as SCIF_No_Atten
# from model.SCIF_reverse_attention import U_Net as SCIF_atten_reverse

# # 与其他模型做比较
from model.c_transunet import TransUNet
# from model.c_wnet import *
# from model.c_Iternet import *
# from model.c_LadderNet import *
# from model.c_BCDUNet import *
# from model.c_fanet import *
# from model.c_kiunet import *
# from model.c_cenet import *
# from model.c_saunet import *
# from model.c_ternaus import *
# from model.c_r2unet import *
# from model.c_multilevel import *
# from model.unet_aspp import UNet as uaspp

from tqdm import tqdm
from utils.helpfunc import *
from torchmetrics.classification import BinaryF1Score,BinaryRecall,BinarySpecificity,BinaryAUROC

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)


def save_checkpoint(root, model, better):
    if better:
        fpath = os.path.join(root, 'best_checkpoint.pth')
    else:
        fpath = os.path.join(root, 'last_checkpoint.pth')
    # torch.save({'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()}, fpath)
    torch.save(model, fpath)

def train(device, model, criterion, optimizer, train_loader, test_loader, epochs, root,model_name):
    model.zero_grad()
    best_loss = 2 ** 16
    running = True
    epoch = 0

    test_writer = SummaryWriter(os.path.join(root, f'{model_name}'))
    
    while epoch <= epochs and running:
        model.train()
        train_loss = 0
        pbar=tqdm(train_loader,colour='#5181D5',desc="Epoch:{}".format(epoch),dynamic_ncols=True,ncols=100)
        for index,(x, y) in enumerate(pbar):
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs)
            # outputs = model([inputs,labels])
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item() / len(train_loader)
            optimizer.step()
            pbar.set_postfix({'loss' : '{0:1.5f}'.format(train_loss)}) #输入一个字典，显示实验指标
            pbar.update(1)


        # 在测试集合上进行验证
        model.eval()
        test_loss = 0
        pixelwize_rank = 0
        pbar_test=tqdm(test_loader,colour='#81D551',desc="testing",dynamic_ncols=True)
        for index,(x, y) in enumerate(pbar_test):
            inputs = x.to(device)
            labels = y.to(device)

            outputs = model(inputs)
            # outputs = model([inputs,labels])
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            save_image(outputs,f'saved_image/{model_name}/{model_name}_{index+1}.png')

            test_loss += loss.item() / len(test_loader)

            spe = BinarySpecificity().to(device)
            f1 = BinaryF1Score().to(device)
            recall=BinaryRecall().to(device)
            auc=BinaryAUROC(thresholds=None).to(device)

            final_clone=outputs.clone()
            final_clone[final_clone>0.5]=1
            final_clone[final_clone<=0.5]=0

            optimizer.step()
            if final_clone.shape == labels.shape:
                pixelwize_rank += int(
                    torch.sum((final_clone > 0.5) == (labels> 0.5))) / labels.data.nelement() * 100 / len(test_loader)
            else:
                pixelwize_rank += int(
                    torch.sum((final_clone.argmax(dim=1) > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(test_loader)


        test_writer.add_scalar(tag='pixel wise accuracy', scalar_value=pixelwize_rank, global_step=epoch)
        test_writer.add_scalar(tag='Sen', scalar_value=recall(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='Spe', scalar_value=spe(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='F1 Score', scalar_value=f1(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='auc', scalar_value=auc(final_clone,labels), global_step=epoch)

        print(f'\
        \n Acc {pixelwize_rank:.4} -\
        \n AUC {auc(final_clone,labels):.4} -\
        \n Se {recall(final_clone,labels):.4} -\
        \n Sp {spe(final_clone,labels):.4} -\
        \n F1 {f1(final_clone,labels):.4}')
        if test_loss < best_loss:
            best_loss = test_loss

        save_checkpoint(root, model, test_loss == best_loss)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_index', help='-1 for cpu', type=int, default=2)
    parser.add_argument('--ckpt_log_dir', type=str, default='ckpt')
    parser.add_argument('--model_name', type=str,default='wnet')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(f'saved_image/{args.model_name}'):
        pass
    else:
        os.makedirs(f'saved_image/{args.model_name}')

    # model = wnet()
    # model = TransUNet(img_dim=512,in_channels=1,out_channels=128,head_num=4,mlp_dim=1024,block_num=6,patch_dim=16,class_num=1)
    # model=LadderNetv6()
    # model=BCDUNet()
    # model = SA_UNet()
    # model=R2U_Net()
    # model=UNet11()
    # model=FCDenseNet(in_channels=1, down_blocks=(4, 4, 4, 4),up_blocks=(4, 4, 4, 4), bottleneck_layers=4,growth_rate=12, out_chans_first_conv=48, n_classes=1)
    # model = FANet()
    # model=kiunet()
    # model=R2U_Net()
    # model=mymodel(class_number=1,in_channels=1)
    # model=FCN(num_classes=1)
    # model=uaspp(n_channels=1,n_classes=1)
    # model=FCDenseNet(in_channels=3, down_blocks=(4, 4, 4, 4),up_blocks=(4, 4, 4, 4), bottleneck_layers=4,growth_rate=12, out_chans_first_conv=48, n_classes=1)
    # model=SCIF_atten_reverse(class_number=1,in_channels=3)
    # model = SCIF_No_Atten(class_number=1,in_channels=3)
    # model=baseline(class_number=1,in_channels=1)
    # model=masp(class_number=1,in_channels=1)
    # model=MINet(class_number=1,in_channels=1)
    # model=stage1()


    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')


    train_dataset=fundus_data('sdnu_eye/train',mode='train')
    test_dataset=fundus_data('sdnu_eye/test',mode='test')
    train_loader=DataLoader(dataset=train_dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=1,num_workers=2,shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    critetions=nn.BCELoss()
    # critetions=Focal_IoU(theta=0.5)

    train(model=model.to(device),
          optimizer=optimizer,
          criterion=critetions.to(device),
          device=device,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=300,
          root=args.ckpt_log_dir,
          model_name=args.model_name
          )
