# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:40:08 2021

@author: ASUS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_backbone import SwinTransformer
from CSConv import CSConv, M_CSConv, CCM

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m, CSConv):
            m._initialize_weights()
        elif isinstance(m, M_CSConv):
            m._initialize_weights()
        elif isinstance(m, CCM):
            m._initialize_weights()
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.LayerNorm):
            pass
        else:
            m.initialize()

class CCLD(nn.Module):
    def __init__(self, swin_type='base', K=3, T=34, channel=64):
        super(CCLD, self).__init__()
        if swin_type == 'base':
            embed_dim=128            
        elif swin_type == 'large':
            embed_dim=192
        
        self.swin_0 = nn.Sequential(nn.Conv2d(embed_dim*8, channel, kernel_size=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.swin_1 = nn.Sequential(nn.Conv2d(embed_dim*4, channel, kernel_size=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.swin_2 = nn.Sequential(nn.Conv2d(embed_dim*2, channel, kernel_size=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.swin_3 = nn.Sequential(nn.Conv2d(embed_dim*1, channel, kernel_size=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        
        self.hint_0 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        
        self.hint_center_0 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_center_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_center_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_center_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        
        self.hint_boundary_0 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_boundary_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_boundary_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.hint_boundary_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
      
        self.CCM_3 = CCM(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CCM_2 = CCM(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CCM_1 = CCM(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CCM_0 = CCM(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        
        self.CCM_center_3  = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_center_2   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_center_1   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_center_0   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_boundary_3   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_boundary_2   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_boundary_1   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CCM_boundary_0   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        
        self.M_CSConv_0 = M_CSConv(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.M_CSConv_1 = M_CSConv(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.M_CSConv_2 = M_CSConv(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.M_CSConv_3 = M_CSConv(channel, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.M_CSConv_center_0   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_center_1   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_center_2   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_center_3   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_boundary_0   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_boundary_1   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_boundary_2   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.M_CSConv_boundary_3   = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        self.CFM_0 = nn.Sequential(nn.Conv2d(channel*3, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_1 = nn.Sequential(nn.Conv2d(channel*3, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_2 = nn.Sequential(nn.Conv2d(channel*3, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_3 = nn.Sequential(nn.Conv2d(channel*3, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        
        self.premask_out = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.CFM_center = CSConv(channel*4, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CFM_center_bn  = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_center_out = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.CFM_boundary = CSConv(channel*4, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CFM_boundary_bn  = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_boundary_out = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.CFM_finalmask = CSConv(channel*4, channel, kernel_size=(3,3), stride=1, padding=1, K=K, T=T)
        self.CFM_finalmask_bn = nn.Sequential(nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.CFM_finalmask_out = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

    def forward(self, input1):
        swin0 = self.swin_0(input1[3])
        swin1 = self.swin_1(input1[2])
        swin2 = self.swin_2(input1[1])
        swin3 = self.swin_3(input1[0])
        
        hint_0 = self.hint_0(swin0)
        hint_1 = swin1 + F.interpolate(hint_0, size=swin1.size()[2:], mode='bilinear')
        hint_1 = self.hint_1(hint_1)
        hint_2 = swin2 + F.interpolate(hint_1, size=swin2.size()[2:], mode='bilinear')
        hint_2 = self.hint_2(hint_2)
        hint_3 = swin3 + F.interpolate(hint_2, size=swin3.size()[2:], mode='bilinear')
        hint_3 = self.hint_3(hint_3)
        premask = self.premask_out(hint_3)

        hint_center_0 = self.hint_center_0(hint_0)
        hint_center_0 = F.interpolate(hint_center_0, size=premask.size()[2:], mode='bilinear')
        hint_center_1 = self.hint_center_1(hint_1)
        hint_center_1 = F.interpolate(hint_center_1, size=premask.size()[2:], mode='bilinear')
        hint_center_2 = self.hint_center_2(hint_2)
        hint_center_2 = F.interpolate(hint_center_2, size=premask.size()[2:], mode='bilinear')
        hint_center_3 = self.hint_center_3(hint_3)
        
        hint_boundary_0 = self.hint_boundary_0(hint_0)
        hint_boundary_0 = F.interpolate(hint_boundary_0, size=premask.size()[2:], mode='bilinear')
        hint_boundary_1 = self.hint_boundary_1(hint_1)
        hint_boundary_1 = F.interpolate(hint_boundary_1, size=premask.size()[2:], mode='bilinear')
        hint_boundary_2 = self.hint_boundary_2(hint_2)
        hint_boundary_2 = F.interpolate(hint_boundary_2, size=premask.size()[2:], mode='bilinear')
        hint_boundary_3 = self.hint_boundary_3(hint_3)

        center_3 = hint_center_3
        boundary_3 = hint_boundary_3
        
        center_3, boundary_3 = self.CCM_3(center_3,boundary_3,premask.sigmoid())
        center_3   = self.CCM_center_3(center_3)
        boundary_3 = self.CCM_boundary_3(boundary_3)
        
        center_2, boundary_2 = self.CCM_2(center_3+hint_center_2,boundary_3+hint_boundary_2,premask.sigmoid())
        center_2   = self.CCM_center_2(center_2)
        boundary_2 = self.CCM_boundary_2(boundary_2)
        
        center_1, boundary_1 = self.CCM_1(center_2+hint_center_1,boundary_2+hint_boundary_1,premask.sigmoid())
        center_1   = self.CCM_center_1(center_1)
        boundary_1 = self.CCM_boundary_1(boundary_1)
        
        center_0, boundary_0 = self.CCM_0(center_1+hint_center_0,boundary_1+hint_boundary_0,premask.sigmoid())
        center_0   = self.CCM_center_0(center_0)
        boundary_0 = self.CCM_boundary_0(boundary_0)
        
        center_0, boundary_0 = self.M_CSConv_0(center_0,boundary_0)
        center_1, boundary_1 = self.M_CSConv_1(center_1,boundary_1)
        center_2, boundary_2 = self.M_CSConv_2(center_2,boundary_2)
        center_3, boundary_3 = self.M_CSConv_3(center_3,boundary_3)
        center_0 = self.M_CSConv_center_0(center_0)
        center_1 = self.M_CSConv_center_1(center_1)
        center_2 = self.M_CSConv_center_2(center_2)
        center_3 = self.M_CSConv_center_3(center_3)
        boundary_0 = self.M_CSConv_boundary_0(boundary_0)
        boundary_1 = self.M_CSConv_boundary_1(boundary_1)
        boundary_2 = self.M_CSConv_boundary_2(boundary_2)
        boundary_3 = self.M_CSConv_boundary_3(boundary_3)
        
        center = self.CFM_center(torch.cat((center_0, center_1, center_2, center_3),dim=1))
        center = self.CFM_center_bn(center)
        center = self.CFM_center_out(center)
        
        boundary = self.CFM_boundary(torch.cat((boundary_0, boundary_1, boundary_2, boundary_3),dim=1))
        boundary = self.CFM_boundary_bn(boundary)
        boundary = self.CFM_boundary_out(boundary)
        
        hint_0_up = F.interpolate(hint_0, size=swin3.size()[2:], mode='bilinear')
        hint_1_up = F.interpolate(hint_1, size=swin3.size()[2:], mode='bilinear')
        hint_2_up = F.interpolate(hint_2, size=swin3.size()[2:], mode='bilinear')
        hint_3_up = F.interpolate(hint_3, size=swin3.size()[2:], mode='bilinear')
        
        finalmask_0 = self.CFM_0(torch.cat((center_0,boundary_0,hint_0_up),dim=1))
        finalmask_1 = self.CFM_1(torch.cat((center_1,boundary_1,hint_1_up),dim=1))
        finalmask_2 = self.CFM_2(torch.cat((center_2,boundary_2,hint_2_up),dim=1))
        finalmask_3 = self.CFM_3(torch.cat((center_3,boundary_3,hint_3_up),dim=1))
        
        finalmask = self.CFM_finalmask_bn(self.CFM_finalmask(torch.cat((finalmask_0, finalmask_1, finalmask_2, finalmask_3),dim=1)))
        finalmask = self.CFM_finalmask_out(finalmask)
        return center, boundary, finalmask, premask
    
    def initialize(self):
        weight_init(self)
      
class CCLDNet(nn.Module):
    def __init__(self, opt=None):
        super(CCLDNet, self).__init__()
        self.opt = opt
        
        self.bkbone   = SwinTransformer(swin_type=opt.swin_type, drop_path_rate=opt.dropout_rate, use_checkpoint=opt.checkpoint)
        if opt.decoder == 'CCLD':
            self.decoder = CCLD(swin_type=opt.swin_type,K=opt.K,T=opt.T)
        
        self.initialize()
        
    def forward(self, x, shape=None):
        out_backbone   = self.bkbone(x)
        center,boundary,finalmask,premask  = self.decoder(out_backbone)
        
        if shape is None:
            shape = x.size()[2:]
        out_premask  = F.interpolate(premask,   size=shape, mode='bilinear')
        out_center  = F.interpolate(center,   size=shape, mode='bilinear')
        out_boundary  = F.interpolate(boundary,   size=shape, mode='bilinear')
        out_finalmask  = F.interpolate(finalmask,   size=shape, mode='bilinear')
        return out_premask, out_center, out_boundary, out_finalmask
        
    def initialize(self):
        if self.opt.model_path is not None:#load pretrained model when testing
            if self.opt.multi_load is True:
                state_dict_multi = torch.load(self.opt.model_path)
                state_dict = {k[7:]: v for k, v in state_dict_multi.items()}
            else:
                state_dict = torch.load(self.opt.model_path)
            self.load_state_dict(state_dict, strict=True)
        else:
            weight_init(self)