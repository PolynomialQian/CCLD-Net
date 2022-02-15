# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
def center_boundary_label(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/mask'):
        mask = cv2.imread(datapath+'/mask/'+name,0)
        center = (cv2.distanceTransform(cv2.blur(mask, ksize=(5,5)), distanceType=cv2.DIST_L2, maskSize=5))**0.5
        center_weight = center/np.max(center)
        mask = center_weight > 0
        center_label = np.floor(mask * center_weight * 255)
        boundary_label = np.floor(mask * (1.0 - center_weight) * 255)
        
        if not os.path.exists(datapath+'/center/'):
            os.makedirs(datapath+'/center/')
        cv2.imwrite(datapath+'/center/'+name, center_label)
        
        if not os.path.exists(datapath+'/boundary/'):
            os.makedirs(datapath+'/boundary/')
        cv2.imwrite(datapath+'/boundary/'+name, boundary_label)
        

if __name__=='__main__':
    center_boundary_label('../data/TrainDataset_Endo')