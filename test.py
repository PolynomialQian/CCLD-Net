#!/usr/bin/python3
#coding=utf-8

import os
import sys
import argparse
from utils.logger import setup_logger
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import dataset
from torch.utils.data import DataLoader
from model import CCLDNet



parser = argparse.ArgumentParser()
parser.add_argument('--multi_load', action='store_true', help='whether to load multi-gpu weight')
parser.add_argument('--swin_type', type=str, default='base', help='tiny,small,base,large')
parser.add_argument('--decoder', type=str, default='CCLD', help='CCLD')
parser.add_argument('--crossnum', type=str, default='endo', help='CA1, CA2, CA3, CA4, CA5, endo, CVC')
parser.add_argument('--K', type=int, default=4, help='the number of anchored weights')
parser.add_argument('--T', type=int, default=31, help='hyperparameter for sparsit')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout_rate in swin_backbone')

parser.add_argument('--model_path', type=str, default='./model_save/EndoScene_CCLDNet/last.pth', help='path to model file')
parser.add_argument('--data_dir', type=str, default='./data/TrainDataset_Endo', help='1,2,3,4,5')
parser.add_argument('--checkpoint', type=int, default=1, help='use_checkpoint')
opt = parser.parse_args()

def get_ACC(SR_real, GT_real):
    acc = 0.0
    GT = GT_real > 0.5
    SR = SR_real >= 0.5
    TP = ((SR == GT) & (GT == 1)).sum(dim=(0,1))
    TN = ((SR == GT) & (GT == 0)).sum(dim=(0,1))
    all_ = (GT>=0).sum(dim=(0,1))
    acc = (TP+TN)/all_
    
    return acc

def get_DC(SR_real, GT_real):
    dice = 0.0
    GT = GT_real > 0.5
    SR = SR_real >= 0.5
    inter = ((SR == GT) & (GT == 1)).sum(dim=(0,1))
    union = SR.sum(dim=(0,1))+GT.sum(dim=(0,1))
    dice = 2.0*inter/union
    
    return dice

def get_Jac(SR_real, GT_real):
    jac = 0.0
    GT = GT_real > 0.5
    SR = SR_real > 0.5
    intersection = ((SR == 1) & (GT == 1)).sum()
    union = ((SR == 1) | (GT == 1)).sum()
    jac = intersection/(union+1e-5)
    return jac

def get_Spe(SR_real, GT_real):
    spe = 0.0
    GT = GT_real > 0.5
    SR = SR_real > 0.5
    intersection = ((SR == 0) & (GT == 0)).sum()
    union = (GT == 0).sum()
    spe = intersection/(union+1e-5)
    return spe

class Test(object):
    def __init__(self, Dataset, Network, Path, opt):
        self.opt=opt
        self.datapath = Path
        if 'CVC_Kva' in Path:
            self.data   = Dataset.CVC_Dataset(Path,trainsize=384, mode='test'+opt.crossnum)
        else:
            self.data   = Dataset.Endo_ISIC_Dataset(Path,trainsize=384, mode='test'+opt.crossnum)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        self.net    = Network(opt)
        self.net.cuda()
        self.net.eval()
   
        
    def save_Endo(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out_premask, out_center, out_boundary, out_finalmask = self.net(image, shape)

                out_finalmask = torch.sigmoid(out_finalmask[0,0]).cpu().numpy()*255
                head = './result/' + self.datapath.split('/')[-1]  + '_' + self.opt.swin_type  + '_' + self.opt.crossnum
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.bmp', np.round(out_finalmask))
    
    def save_2018(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out_premask, out_center, out_boundary, out_finalmask = self.net(image, shape)

                out_finalmask = torch.sigmoid(out_finalmask[0,0]).cpu().numpy()*255
                head = './result/' + self.datapath.split('/')[-1]  + '_' + self.opt.swin_type + '_' + self.opt.crossnum
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'_segmentation.png', np.round(out_finalmask))
    
    def save_CVC_Kva(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out_premask, out_center, out_boundary, out_finalmask = self.net(image, shape)

                out_finalmask = torch.sigmoid(out_finalmask[0,0]).cpu().numpy()*255
                head = './result/' + self.datapath.split('/')[-1]  + '_' + self.opt.swin_type + '_' + self.opt.crossnum
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0], np.round(out_finalmask))
                
    def score(self):
        finalmask_path = './result/' + self.datapath.split('/')[-1]  + '_' + self.opt.swin_type  + '_' + self.opt.crossnum
        gtpath = self.datapath + '/mask'
        num = len(os.listdir(finalmask_path))
        dice = 0.0
        jac = 0.0
        spe = 0.0
        acc = 0.0
        
        for name in os.listdir(finalmask_path):
            pre  = cv2.imread(finalmask_path + '/' + name, 0)
            pre = pre/255
            pre  = torch.from_numpy(pre)
            
            gt  = cv2.imread(gtpath + '/' + name, 0)
            gt = gt/255
            gt  = torch.from_numpy(gt)
            
            dice   += get_DC(pre, gt)
            acc    += get_ACC(pre, gt)
            spe    += get_Spe(pre, gt)
            jac    += get_Jac(pre, gt)
            
          
        logger.info("{}:\n dice:{}\n acc:{}\n spe:{}\n jac:{}".format(self.datapath.split('/')[-1], dice/num, acc/num, spe/num, jac/num))
        
if __name__=='__main__':
    os.makedirs('./model_save/' + opt.model_path.split('/')[-2], exist_ok=True) 
    logger = setup_logger(output='./model_save/' + opt.model_path.split('/')[-2], name="CCLDNet")  
    path = opt.data_dir
    if 'Endo' in path:
        t = Test(dataset, CCLDNet, path, opt)
        t.save_Endo()
        t.score()
    elif '2018' in path:
        t = Test(dataset, CCLDNet, path, opt)
        t.save_2018()
        t.score()
    elif 'CVC_Kva' in path:
        for dataset_path in [path+'/CVC-ClinicDB', path+'/Kvasir', path+'/ETIS-LaribPolypDB', path+'/CVC-ColonDB', path+'/CVC-300']:
            t = Test(dataset, CCLDNet, dataset_path, opt)
            t.save_CVC_Kva()
            t.score()
    
       