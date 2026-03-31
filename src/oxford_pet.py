import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(512, 512)):
        self.root_dir = root_dir
        self.split = split
        self.size = size
        
        # 讀取名單
        split_file = os.path.join(root_dir, 'annotations', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.file_names = [line.split()[0] for line in f]

        # 基礎轉換：Resize -> ToTensor -> Normalize
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_names)


    
    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root_dir, 'images', f'{img_name}.jpg')
        mask_path = os.path.join(self.root_dir, 'annotations', 'trimaps', f'{img_name}.png')
        
        # 1. 先確保圖片和遮罩被讀取（定義變數）
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # 2. 標籤處理
        mask_np = np.array(mask)
        mask_target = (mask_np == 1).astype(np.float32)
        mask = Image.fromarray(mask_target)

        # 3. 只有訓練集才做 Data Augmentation
        if self.split == 'train':
            # 水平翻轉
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # 隨機旋轉
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
                
            # 色彩抖動 (只對 image)
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
                image = color_jitter(image)

        # 4. 最後回傳（這兩行必須跟上面的 image = ... 對齊，不能縮進到 if 裡面）
        return self.img_transform(image), self.mask_transform(mask)