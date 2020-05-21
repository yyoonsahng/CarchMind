from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        '''
            csv_file : 데이터셋 관련 정보 들어있는 csv 파일
            root_dir : 이미지 들어있는 폴더 경로
            transform : 이미지에 적용할 transform
        '''
        self.car_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.car_frame)
    
    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.car_frame.iloc[idx, 0])
        image = Image.open(img_name)
        
        #이미지에서 자동차 부분만 crop
        ground_box = []
        for i in range(1, 5):
            ground_box.append(self.car_frame.iloc[idx, i])    
        image = image.crop(tuple(ground_box))
        
        #비율 유지하기 위해서 padding 추가
        w, h = image.size
        padding = 0
        out = image
        if w > h:
            padding = int((w - h) / 2)
            width = w
            height = h + 2 * padding
            out = Image.new("RGB", (width, height), '#FFFFFF')
            out.paste(image, (0, padding))
        else:
            padding = int((h - w) / 2)
            width = w + 2 * padding
            height = h
            out = Image.new("RGB", (width, height), '#FFFFFF')
            out.paste(image, (padding, 0))
            
        image = out
        label = self.car_frame.iloc[idx, 5]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return (image, label)