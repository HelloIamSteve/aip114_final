import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from config import *

class Lunch500(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', transform=transforms.ToTensor()):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.mode_dir = os.path.join(root_dir, mode) # use mode to split dataset
        self.labels = os.listdir(self.mode_dir)

        self.info = []
        for i, label in enumerate(self.labels):
            for filename in os.listdir(os.path.join(self.mode_dir, label)):
                self.info.append((os.path.join(self.mode_dir, label, filename),
                                  i))

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        # return the label and the image in rgb order
        filename, label = self.info[idx]
        # img = cv2.imread(filename)
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
if __name__ == '__main__':
    dataset_dir = os.path.join('lunch500')
    lunch500_dataset = Lunch500(root_dir=dataset_dir, transform=transforms.ToTensor())
    print(f'Dataset size: {len(lunch500_dataset)}')

    test_idx = 1500
    img, label = lunch500_dataset[test_idx]
    print(f'img.shape: {img.shape}, label: {label}')

    img = (img.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
    plt.imsave(f'{label_names[label]}.png', img)