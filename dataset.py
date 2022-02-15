# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import ImageEnhance
import pickle
from PIL import Image
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import cv2
def cv_random_hflip(image, mask, center, boundary):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        center = center.transpose(Image.FLIP_LEFT_RIGHT)
        boundary = boundary.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask, center, boundary

def cv_random_vflip(image, mask, center, boundary):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        center = center.transpose(Image.FLIP_TOP_BOTTOM)
        boundary = boundary.transpose(Image.FLIP_TOP_BOTTOM)
    return image, mask, center, boundary

def randomCrop(image, mask, center, boundary):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), mask.crop(random_region),center.crop(random_region),boundary.crop(random_region)
def randomRotation(image,mask,center,boundary):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        mask=mask.rotate(random_angle, mode)
        center=center.rotate(random_angle, mode)
        boundary=boundary.rotate(random_angle, mode)
    return image,mask,center,boundary
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)






class Endo_ISIC_Dataset(data.Dataset):
    def __init__(self, data_root, mode='train', trainsize=224, hflip=False, vflip=False):
        self.trainsize = trainsize
        self.hflip = hflip
        self.vflip = vflip
        self.mode = mode
        with open(data_root+'/'+ mode +'.txt', 'r') as lines:
            self.images = []
            self.masks = []
            self.centers = []
            self.boundarys = []
            self.names = []
            if 'Endo' in data_root:
                for line in lines:
                    self.images.append(data_root + '/image/' + line.strip() + '.bmp')
                    self.masks.append(data_root + '/mask/' + line.strip() + '.bmp')
                    self.centers.append(data_root + '/center/' + line.strip() + '.bmp')
                    self.boundarys.append(data_root + '/boundary/' + line.strip() + '.bmp')
                    self.names.append(line.strip())
            elif 'CVC_Kva' in data_root:
                for line in lines:
                    self.images.append(data_root + '/image/' + line.strip() + '.png')
                    self.masks.append(data_root + '/mask/' + line.strip() + '.png')
                    self.centers.append(data_root + '/center/' + line.strip() + '.png')
                    self.boundarys.append(data_root + '/boundary/' + line.strip() + '.png')
                    self.names.append(line.strip())
            elif '2018' in data_root:
                for line in lines:
                    self.images.append(data_root + '/image/' + line.strip().replace('.npy', '.jpg'))
                    self.masks.append(data_root + '/mask/' + line.strip().replace('.npy', '_segmentation.png'))
                    self.centers.append(data_root + '/center/' + line.strip().replace('.npy', '_segmentation.png'))
                    self.boundarys.append(data_root + '/boundary/' + line.strip().replace('.npy', '_segmentation.png'))
                    self.names.append(line.strip().replace('.npy', ''))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.centers = sorted(self.centers)
        self.boundarys = sorted(self.boundarys)
        self.names = sorted(self.names)
        self.filter_files()
        self.size = len(self.images)
        filename = os.path.join(data_root, 'processed_' + mode + '_data.pkl')
        if not os.path.exists(filename):
            images = []
            masks = []
            centers = []
            boundarys = []
            for i in range(self.size):
                image = self.rgb_loader(self.images[i])
                mask = self.binary_loader(self.masks[i])
                center = self.binary_loader(self.centers[i])
                boundary = self.binary_loader(self.boundarys[i])
                images.append(image)
                masks.append(mask)
                centers.append(center)
                boundarys.append(boundary)
            self.image_data = images
            self.mask_data = masks
            self.center_data = centers
            self.boundary_data = boundarys
            with open(filename, 'wb') as f:
                pickle.dump((self.image_data, self.mask_data, self.center_data, self.boundary_data), f)
            print(f"data saved in {filename}")
        else:
            print(f"data loaded in {filename}")
            with open(filename, 'rb') as f:
                self.image_data, self.mask_data, self.center_data, self.boundary_data = pickle.load(f)

        self.image_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_transform_new = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.center_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])
        self.boundary_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = self.image_data[index]
        mask = self.mask_data[index]
        center = self.center_data[index]
        boundary = self.boundary_data[index]
        name = self.names[index]
        if 'train' in self.mode:
            if self.hflip:
                image, mask, center, boundary = cv_random_hflip(image, mask, center, boundary)
            if self.vflip:
                image, mask, center, boundary = cv_random_vflip(image, mask, center, boundary)
            image, mask, center, boundary =randomCrop(image, mask, center, boundary)
            image, mask, center, boundary = randomRotation(image, mask, center, boundary)
            image = colorEnhance(image)
            mask = randomPeper(mask)
            image = (image-np.array([[[124.55, 118.90, 102.94]]]))/np.array([[[ 56.77,  55.97,  57.50]]])
            mask = mask/np.array([255])
            center = center/np.array([255])
            boundary = boundary/np.array([255])
            return image, mask, center, boundary
        if 'val' in self.mode:
            shape = image.size[::-1]
            image = self.image_transform(image)
            mask = transforms.ToTensor()(mask)
            return image, mask, shape, name
        if 'test' in self.mode:
            shape = image.size[::-1]
            image = self.image_transform(image)
            return image, shape, name

    def filter_files(self):
        assert len(self.images) == len(self.masks)
        images = []
        masks = []
        centers = []
        boundarys = []
        for img_path, mask_path, center_path, boundary_path in zip(self.images, self.masks, self.centers, self.boundarys):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            center = Image.open(center_path)
            boundary = Image.open(boundary_path)
            if img.size == mask.size:
                images.append(img_path)
                masks.append(mask_path)
            if center.size == mask.size:
                centers.append(center_path)
            if boundary.size == mask.size:
                boundarys.append(boundary_path)

        self.images = images
        self.masks = masks
        self.centers = centers
        self.boundarys = boundarys

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def __len__(self):
        return self.size

    def collate(self, batch):
        scale = int((np.random.rand(1) + 0.5) * 384)
        image, mask, center, boundary = [list(item) for item in zip(*batch)]
        padded_image = torch.zeros((len(batch), 384, 384, 3))
        padded_mask = torch.zeros((len(batch), 384, 384, 1))
        padded_center = torch.zeros((len(batch), 384, 384, 1))
        padded_boundary = torch.zeros((len(batch), 384, 384, 1))
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            for i in range(len(batch)): 
                image[i] = cv2.resize(image[i], dsize=(scale, 384), interpolation=cv2.INTER_LINEAR)
                mask[i] = cv2.resize(mask[i], dsize=(scale, 384), interpolation=cv2.INTER_LINEAR)
                center[i] = cv2.resize(center[i], dsize=(scale, 384), interpolation=cv2.INTER_LINEAR)
                boundary[i] = cv2.resize(boundary[i], dsize=(scale, 384), interpolation=cv2.INTER_LINEAR)
                if scale <= 384:
                    padded_image[i, 0:384,0:scale] = torch.from_numpy(image[i])
                    padded_mask[i, 0:384,0:scale] = torch.from_numpy(mask[i]).unsqueeze(2)
                    padded_center[i, 0:384,0:scale] = torch.from_numpy(center[i]).unsqueeze(2)
                    padded_boundary[i, 0:384,0:scale] = torch.from_numpy(boundary[i]).unsqueeze(2)
                else:
                    padded_image[i, 0:384,0:384] = torch.from_numpy(image[i])[:,0:384,:]
                    padded_mask[i, 0:384,0:384] = torch.from_numpy(mask[i]).unsqueeze(2)[:,0:384,:]
                    padded_center[i, 0:384,0:384] = torch.from_numpy(center[i]).unsqueeze(2)[:,0:384,:]
                    padded_boundary[i, 0:384,0:384] = torch.from_numpy(boundary[i]).unsqueeze(2)[:,0:384,:]
        else:
            for i in range(len(batch)): 
                image[i] = cv2.resize(image[i], dsize=(384, scale), interpolation=cv2.INTER_LINEAR)
                mask[i] = cv2.resize(mask[i], dsize=(384, scale), interpolation=cv2.INTER_LINEAR)
                center[i] = cv2.resize(center[i], dsize=(384, scale), interpolation=cv2.INTER_LINEAR)
                boundary[i] = cv2.resize(boundary[i], dsize=(384, scale), interpolation=cv2.INTER_LINEAR)
                if scale <= 384:
                    padded_image[i, 0:scale,0:384] = torch.from_numpy(image[i])
                    padded_mask[i, 0:scale,0:384] = torch.from_numpy(mask[i]).unsqueeze(2)
                    padded_center[i, 0:scale,0:384] = torch.from_numpy(center[i]).unsqueeze(2)
                    padded_boundary[i, 0:scale,0:384] = torch.from_numpy(boundary[i]).unsqueeze(2)
                else:
                    padded_image[i, 0:384,0:384] = torch.from_numpy(image[i])[0:384,:,:]
                    padded_mask[i, 0:384,0:384] = torch.from_numpy(mask[i]).unsqueeze(2)[0:384,:,:]
                    padded_center[i, 0:384,0:384] = torch.from_numpy(center[i]).unsqueeze(2)[0:384,:,:]
                    padded_boundary[i, 0:384,0:384] = torch.from_numpy(boundary[i]).unsqueeze(2)[0:384,:,:]
        
        return padded_image.permute(0,3,1,2), padded_mask.permute(0,3,1,2), padded_center.permute(0,3,1,2), padded_boundary.permute(0,3,1,2)






class CVC_Dataset(data.Dataset):
    def __init__(self, data_root, mode='train', trainsize=224, hflip=False, vflip=False):
        self.trainsize = trainsize
        self.hflip = hflip
        self.vflip = vflip
        self.mode = mode
        self.images = []
        self.names = []
        for filename in os.listdir(data_root+ '/image'):
            self.images.append(data_root + '/image/' + filename)
            self.names.append(filename)
        self.images = sorted(self.images)
        self.names = sorted(self.names)
        self.size = len(self.images)
        
        images = []
        for i in range(self.size):
            image = self.rgb_loader(self.images[i])
            images.append(image)
        self.image_data = images
            
        self.image_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        image = self.image_data[index]
        name = self.names[index]
        if 'test' in self.mode:
            shape = image.size[::-1]
            image = self.image_transform(image)
            return image, shape, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __len__(self):
        return self.size