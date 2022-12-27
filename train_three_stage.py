import os
import torch
import argparse
from utils.data_loader import fundus_data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from utils.new_loss import *
from tqdm import tqdm
from utils.helpfunc import *
from utils.new_loss import Focal_IoU
from torchmetrics.classification import BinaryF1Score,BinaryJaccardIndex,BinaryRecall,BinaryPrecision,BinarySpecificity,BinaryAUROC
from model.MINet import MINet as mymodel
from model.stage1 import refinement,stage1
from utils.save_model import *



def train(device, model1,model2,model3,criterion1,criterion2,criterion3, optimizer1,optimizer2,optimizer3, train_loader, test_loader, epochs, root):
    model1.zero_grad()
    model2.zero_grad()
    model3.zero_grad()
    best_loss = 2 ** 16
    running = True
    epoch = 0
    test_writer = SummaryWriter(os.path.join(root, 'mymodel'))

    while epoch <= epochs and running:
        model1.train()
        model2.train()
        model3.train()
        train_loss = 0
        pbar = tqdm(train_loader, colour='#5181D5', desc="Epoch:{}".format(epoch), dynamic_ncols=True, ncols=100)
        for index, (x, y) in enumerate(pbar):
            inputs, labels = x.to(device), y.to(device)

            first_pred_mask=model1(inputs)
            save_image(first_pred_mask, f'saved_image/first_pred/train/first_pred_mask_{index + 1}.png')
            loss_model1 = criterion1(first_pred_mask, labels)

            first_pred_mask_clone=first_pred_mask.clone()
            first_pred_mask_clone[first_pred_mask_clone>0.7]=1
            first_pred_mask_clone[first_pred_mask_clone<=0.7]=0
            gt_model2=labels-first_pred_mask_clone#将第一个网络的分割结果二值化之后与labels相减的到第二个model的labels
            res=labels-first_pred_mask.clone()#这是第一个网络分割不理想的部分
            model2_input=torch.cat([inputs,res],dim=1)

            second_pred_mask=model2(model2_input)
            save_image(torch.cat([second_pred_mask,gt_model2],dim=0),f'saved_image/second_pred/train/final_mask_{index + 1}.png')
            save_image(gt_model2,f'saved_image/second_pred/train/gt_model2_{index + 1}.png')
            loss_model2=criterion2(second_pred_mask,gt_model2)

            model3_input=torch.add(first_pred_mask.clone(),second_pred_mask)
            final_mask=model3(model3_input)
            save_image(torch.cat([final_mask,labels],dim=0),f'saved_image/final_pred/train/final_mask_{index + 1}.png')
            loss_model3=criterion3(final_mask,labels)

            # 总loss
            total_loss=loss_model1+loss_model2+loss_model3

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            train_loss += total_loss.item() / len(train_loader)

            pbar.set_postfix({'loss': '{0:1.5f}'.format(train_loss)})  # 输入一个字典，显示实验指标
            pbar.update(1)


        # 在测试集合上进行验证
        model1.eval()
        model2.eval()
        model3.eval()
        test_loss = 0
        pixelwize_rank = 0
        pbar_test = tqdm(test_loader, colour='#81D551', desc="testing", dynamic_ncols=True)
        for index, (x, y) in enumerate(pbar_test):
            inputs, labels = x.to(device), y.to(device)

            first_pred_mask=model1(inputs)
            save_image(first_pred_mask, f'saved_image/first_pred/test/first_pred_mask_{index + 1}.png')
            loss_model1 = criterion1(first_pred_mask, labels)

            first_pred_mask_clone=first_pred_mask.clone()
            first_pred_mask_clone[first_pred_mask_clone>0.7]=1
            first_pred_mask_clone[first_pred_mask_clone<=0.7]=0
            gt_model2=labels-first_pred_mask_clone#将第一个网络的分割结果二值化之后与labels相减的到第二个model的labels
            res=labels-first_pred_mask.clone()#这是第一个网络分割不理想的部分
            model2_input=torch.cat([inputs,res],dim=1)

            second_pred_mask=model2(model2_input)
            save_image(torch.cat([second_pred_mask,gt_model2],dim=0),f'saved_image/second_pred/test/final_mask_{index + 1}.png')
            loss_model2=criterion2(second_pred_mask,gt_model2)

            model3_input=torch.add(first_pred_mask.clone(),second_pred_mask)
            final_mask=model3(model3_input)
            save_image(torch.cat([final_mask,labels],dim=0),f'saved_image/final_pred/test/final_mask_{index + 1}.png')
            loss_model3=criterion3(final_mask,labels)

            # 总loss
            total_loss=loss_model1+loss_model2+loss_model3

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            final_clone=final_mask.clone()
            final_clone[final_clone>0.7]=1
            final_clone[final_clone<=0.7]=0

            test_loss+=total_loss.item()/len(test_loader)

            spe = BinarySpecificity().to(device)
            f1 = BinaryF1Score().to(device)
            iou=BinaryJaccardIndex().to(device)
            recall=BinaryRecall().to(device)
            pre=BinaryPrecision().to(device)
            auc=BinaryAUROC(thresholds=None).to(device)

            if final_clone.shape == labels.shape:
                pixelwize_rank += int(
                    torch.sum((final_clone > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(test_loader)
            else:
                pixelwize_rank += int(
                    torch.sum((final_clone.argmax(dim=1) > 0.5) == (labels > 0.5))) / labels.data.nelement() * 100 / len(
                    test_loader)



        test_writer.add_scalar(tag='pixel wise accuracy', scalar_value=pixelwize_rank, global_step=epoch)
        test_writer.add_scalar(tag='Sen', scalar_value=recall(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='Spe', scalar_value=spe(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='F1 Score', scalar_value=f1(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='auc', scalar_value=auc(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='iou', scalar_value=iou(final_clone,labels), global_step=epoch)
        test_writer.add_scalar(tag='pre', scalar_value=pre(final_clone,labels), global_step=epoch)

        print(f'\
        \n Acc {pixelwize_rank:.4} -\
        \n AUC {auc(final_clone,labels):.4} -\
        \n Se {recall(final_clone,labels):.4} -\
        \n Sp {spe(final_clone,labels):.4} -\
        \n F1 {f1(final_clone,labels):.4}')

        if test_loss < best_loss:
            best_loss = test_loss

        save_checkpoint(root, model1,model2, test_loss == best_loss)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_index', help='-1 for cpu', type=int, default=2)
    parser.add_argument('--ckpt_log_dir', type=str, default='ckpt')
    parser.add_argument('--train_data_dir', type=str, default='sdnu_eye/train')
    parser.add_argument('--test_data_dir', type=str, default='sdnu_eye/test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_works', type=int, default=4)
    args = parser.parse_args()
    print(args)

    # 第二个模块的输入包括三个部分，Image，pred mask 以及 第一个模块没有分割完全的剩余部分 res mask
    first_model=stage1()
    second_model=mymodel(class_number=1,in_channels=2)
    refine=refinement()

    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')

    train_dataset = fundus_data(args.train_data_dir, mode='train')
    test_dataset = fundus_data(args.test_data_dir, mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_works, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_works, shuffle=False)

    optimizer1 = torch.optim.Adam(first_model.parameters(), lr=1e-4,betas=(0.9, 0.999))
    optimizer2 = torch.optim.Adam(second_model.parameters(), lr=1e-4,betas=(0.9, 0.999))
    optimizer3 = torch.optim.Adam(second_model.parameters(), lr=1e-4,betas=(0.9, 0.999))

    critetions1 = nn.BCELoss()
    critetions2 = Focal_IoU(theta=0.5)
    critetions3 = nn.BCELoss()

    

    train(model1=first_model.to(device),
          model2=second_model.to(device),
          model3=refine.to(device),
          optimizer1=optimizer1,
          optimizer2=optimizer2,
          optimizer3=optimizer3,
          criterion1=critetions1.to(device),
          criterion2=critetions2.to(device),
          criterion3=critetions3.to(device),
          device=device,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=300,
          root=args.ckpt_log_dir
          )
