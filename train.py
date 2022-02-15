# -*- coding: utf-8 -*-
import sys
import time
import os
import json
import argparse
from torch.autograd import Variable
from utils.logger import setup_logger
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model  import CCLDNet
from utils.lr_scheduler import get_scheduler

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def parse_option():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--crossnum', type=str, default='endo', help='CA1, CA2, CA3, CA4, CA5, endo, CVC')
    # training
    parser.add_argument('--hflip', type=bool,default=True, help='hflip data')
    parser.add_argument('--vflip', type=bool,default=True, help='vflip data')
    parser.add_argument('--checkpoint', type=int, default=1, help='use_checkpoint')
    parser.add_argument('--ratio', type=float, default=1, help='hidden_number/input_number in the linear_layer')
    parser.add_argument('--K', type=int, default=4, help='the number of weights')
    parser.add_argument('--T', type=int, default=31, help='hyperparameter for sparsity')
    parser.add_argument('--swin_type', type=str, default='base', help='base,large')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout_rate in swin_backbone')
    parser.add_argument('--decoder', type=str, default='CCLD', help='CCLD')
    parser.add_argument('--lr', type=float, default=0.8, help='learning rate')
    parser.add_argument('--lr_rate', type=float, default=0.1, help='lr_backbone/lr_head') 
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--optim', type=str, default='SGD', help="SGD,AdamW")
    parser.add_argument('--head_weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--backbone_weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    
    parser.add_argument('--model_path', type=str, default=None, help='model path to pretrain')
    parser.add_argument('--output_dir', type=str, default='./model_save', help='output director')
    parser.add_argument('--data_dir', type=str, default='./data/TrainDataset_Endo', help='./data/TrainDataset_Endo, ./data/TrainDataset_CVC_Kva, ./data/TrainDataset_2018')
    opt, unparsed = parser.parse_known_args()
    opt.output_dir = os.path.join(opt.output_dir, str(int(time.time())) + '_' + opt.decoder + '_' + opt.swin_type + '_' + opt.crossnum)
    return opt

def get_DC(SR_real, GT):
    # DC : Dice Coefficient
    GT = GT > 0.5
    dice_sum = 0.0
    for i in range(0,1):
        SR = SR_real >= 0.5
        inter = ((SR == GT) & (GT == 1)).sum(dim=(1,2,3))
        union = SR.sum(dim=(1,2,3))+GT.sum(dim=(1,2,3))
        dice_sum += 2.0*inter/union
    
    return dice_sum

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/((union-inter+1)+1e-6)
    return iou.mean()

def build_loader(opt):
    num_gpus = torch.cuda.device_count()
    print("========>num_gpus:{}==========".format(num_gpus))

    train_data   = dataset.Endo_ISIC_Dataset(opt.data_dir, mode='train'+opt.crossnum, trainsize=384, hflip=opt.hflip, vflip=opt.vflip)
    train_loader = DataLoader(train_data,collate_fn=train_data.collate, batch_size=opt.batchsize*num_gpus, shuffle=True, pin_memory=False, num_workers=8)
    val_data   = dataset.Endo_ISIC_Dataset(opt.data_dir, trainsize=384,mode='val'+opt.crossnum)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    return train_loader, val_loader

def build_model(opt):
    # build model
    Network = CCLDNet(opt)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        Network = nn.DataParallel(Network)
    Network = Network.cuda()
    return Network

def main(opt):
    
    train_loader, val_loader = build_loader(opt)
    Network = build_model(opt)
    base, head = [], []
    # build optimizer
    for name, param in Network.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)     
    if opt.optim == 'SGD':
        optimizer      = torch.optim.SGD([{'params':base, 'weight_decay': opt.backbone_weight_decay, 'lr': opt.lr * opt.lr_rate * opt.batchsize/128}, {'params':head}], lr=opt.lr*opt.batchsize/128, momentum=opt.momentum, weight_decay=opt.head_weight_decay, nesterov=True)
    elif opt.optim == 'AdamW':
        optimizer      = torch.optim.AdamW([{'params':base, 'weight_decay': opt.head_weight_decay, 'lr': opt.lr * opt.lr_rate * opt.batchsize/128}, {'params':head}], lr=opt.lr*opt.batchsize/128, weight_decay=opt.head_weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    if opt.decoder == 'CCLD':
        for epoch in range(opt.epoch):
            train(train_loader, Network, opt, epoch, optimizer, scheduler)
            validate(val_loader, Network, opt, epoch)
    return os.path.join(opt.output_dir, "last_epoch.pth")
       
def train(data_loader, Network, opt, epoch, optimizer, scheduler):
    net    = Network
    net.train(True)
    global_step    = 0
    num = len(data_loader)
    loss_finalmask_all = 0.0
    loss_center_all = 0.0
    loss_boundary_all = 0.0
    loss_premask_all = 0.0
    # routine
    for step, (image, mask, center, boundary) in enumerate(data_loader):
        optimizer.zero_grad()
        image, mask, center, boundary = Variable(image).cuda(), Variable(mask).cuda(), Variable(center).cuda(), Variable(boundary).cuda()
        out_premask, out_center, out_boundary, out_finalmask = net(image)

        loss_premask  = F.binary_cross_entropy_with_logits(out_premask, mask) + iou_loss(out_premask, mask)
        loss_center  = F.binary_cross_entropy_with_logits(out_center, center) 
        loss_boundary  = F.binary_cross_entropy_with_logits(out_boundary, boundary)  
        loss_finalmask  = F.binary_cross_entropy_with_logits(out_finalmask, mask) + iou_loss(out_finalmask, mask)
        
        
        loss   = (loss_premask + loss_center +  loss_boundary + loss_finalmask)/4
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        scheduler.step()
        
        loss_premask_all += loss_premask.item()
        loss_center_all += loss_center.item()
        loss_boundary_all += loss_boundary.item()
        loss_finalmask_all += loss_finalmask.item()

        global_step += 1
    sw.add_scalars('lr'   , {'lr_backbone':optimizer.param_groups[0]['lr'], 'lr_head':optimizer.param_groups[0]['lr']}, global_step=epoch+1)
    sw.add_scalars('trainloss', {'premask':loss_premask_all/num, 'center':loss_center_all/num, 'boundary':loss_boundary_all/num, 'mask':loss_finalmask_all/num}, global_step=epoch+1)
    logger.info('step:%d/%d/%d | lr_backbone=%.6f | lr_head=%.6f | loss_premask_ave=%.6f | loss_center_ave=%.6f | loss_boundary_ave=%.6f | loss_finalmask_ave=%.6f'
                %(global_step, epoch+1, opt.epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'], loss_premask_all/num, loss_center_all/num, loss_boundary_all/num, loss_finalmask_all/num))
    if (epoch+1) % 10 == 0:
        torch.save(net.state_dict(), os.path.join(opt.output_dir, "epoch_{}_ckpt.pth".format(epoch + 1)))
        logger.info("model saved {}!, learning_rate {}".format(os.path.join(opt.output_dir, "epoch_{}_ckpt.pth".format(epoch + 1)), optimizer.param_groups[0]['lr']))
    if (epoch+1)  == opt.epoch:
        torch.save(net.state_dict(), os.path.join(opt.output_dir, "last_epoch.pth"))
        logger.info("last_model saved !")
    
def validate(data_loader, Network, opt, epoch):
    net = Network
    net.eval()
    num = len(data_loader)
    sum_loss = 0.0
    sum_loss_pre = 0.0
    dice_score = 0.0
    dice_score_pre = 0.0
    with torch.no_grad():
        for image, mask, (H, W), name in data_loader:
            image, shape  = image.cuda().float(), (H, W)
            mask = mask.cuda().float()
            
            out_premask, out_center, out_boundary, out_finalmask = net(image, shape)

            loss = F.binary_cross_entropy_with_logits(out_finalmask, mask) + iou_loss(out_finalmask, mask)
            sum_loss += loss.item()
            loss_pre = F.binary_cross_entropy_with_logits(out_premask, mask) + iou_loss(out_premask, mask)
            sum_loss_pre += loss_pre.item()
            
            out_finalmask_score = torch.nn.Sigmoid()(out_finalmask)
            dice_score += get_DC(out_finalmask_score, mask)
            out_premask_score = torch.nn.Sigmoid()(out_premask)
            dice_score_pre += get_DC(out_premask_score, mask)

        sw.add_scalars('testloss', {'test_final':sum_loss/num, 'test_pre':sum_loss_pre/num}, global_step=epoch+1)
        logger.info('epoch:%d | testloss=%.6f | testloss_pre=%.6f | dice=%.4f | dice_pre=%.4f' 
                %(epoch+1, sum_loss/num, sum_loss_pre/num, dice_score/num, dice_score_pre/num))
        

if __name__=='__main__':
    best_dice = 0.0
    opt = parse_option()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    
    sw = SummaryWriter(opt.output_dir)
    logger = setup_logger(output=opt.output_dir, name="CCLDNet")
    path = os.path.join(opt.output_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    ckpt_path = main(opt)
    