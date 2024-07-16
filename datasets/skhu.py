import pandas as pd
import torch
from natsort import natsorted
import glob
import random
import os
import numpy as np
from torchvision import transforms
from PIL import Image


class SKHU(torch.utils.data.Dataset):
    def __init__(self, config, data_path):
        self.window = config.window
        self.img_h = config.img_h
        self.img_w = config.img_w
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.images = glob.glob(os.path.join(self.data_path, "*/*.jpg"))
        self.sections = [(f.path, f.name) for f in os.scandir(data_path) if f.is_dir()]
        self.maxnum = {}
        for path, name in self.sections:
            files = natsorted(glob.glob(os.path.join(path, "*.jpg")))
            first_filename = os.path.splitext(os.path.basename(files[0]))[0]
            last_filename = os.path.splitext(os.path.basename(files[-1]))[0]
            self.maxnum[name] = [int(first_filename[6:]), int(last_filename[6:])]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anc_path = self.images[idx]
        pos_path = self.get_positive(anc_path)
        neg_path = self.get_negative(anc_path)
        anc = Image.open(anc_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        anc = self.transform(anc)
        pos = self.transform(pos)
        neg = self.transform(neg)

        return anc, pos, neg
    
    
        
    def get_positive(self, path):
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0][6:])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            number_dst = random.randrange(max(self.maxnum[section_src][0], number_src-self.window), min(number_src+self.window+1, self.maxnum[section_src][1]))
            if number_src != number_dst:
                break

        path = path.replace(makefour(number_src)+ext, makefour(number_dst)+ext)

        return path

    def get_negative(self, path):
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0][6:])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            section_dst = random.choice(self.sections)[1]
            number_dst = random.randrange(self.maxnum[section_dst][0], self.maxnum[section_dst][1]+1)
            if abs(number_src - number_dst) > self.window:
                break

        #path = path.replace(section_src, section_dst)
        path = path.replace(makefour(number_src)+ext, makefour(number_dst)+ext)

        return path
    
def makefour(n):
        sn=str(n)
        l=len(sn)
        if l==1:
            return '000'+sn
        elif l==2:
            return '00'+sn
        elif l==3:
            return '0'+sn
        else:
            return sn