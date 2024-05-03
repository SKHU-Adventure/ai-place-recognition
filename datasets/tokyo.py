import torch
from natsort import natsorted
import glob
import random
import os
import numpy as np
from skimage import io
from torchvision import transforms
from PIL import Image

class Tokyo(torch.utils.data.Dataset): 

    def __init__(self, config, data_path):
        self.window = config.window
        self.img_h = config.img_h
        self.img_w = config.img_w
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.data_path = data_path
        self.img_list = glob.glob(os.path.join(self.data_path, "*/*/*.png"))
        self.img_list = natsorted(self.img_list)
        self.pos_list = []
        self.transform = transforms.Compose([
                transforms.Resize((config.img_h, config.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self._generate_pos_list()

    def _generate_pos_list(self):
        for idx, img in enumerate(self.img_list):
            temp = []
            current_angle = int(img.split("_")[-1].split(".")[0])
            positive_angles = [(current_angle - 30) % 360, (current_angle + 30) % 360]

            for other_img in self.img_list:
                if other_img != img and os.path.dirname(img) == os.path.dirname(other_img):
                    other_angle = int(other_img.split("_")[-1].split(".")[0])
                    if other_angle in positive_angles:
                        temp.append(other_img)
            self.pos_list.append(temp)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx): 
        anc_path = self.img_list[idx]
        pos_path = np.random.choice(self.pos_list[idx])
        while True:
            i = random.randrange(len(self.img_list))
            if (i != idx) and (self.img_list[i] not in self.pos_list[idx]):
                neg_path = self.img_list[i]
                break
        anc = Image.open(anc_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')

        if self.transform:
            anc = self.transform(anc)
            pos = self.transform(pos)
            neg = self.transform(neg)
        return anc, pos, neg