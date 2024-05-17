import pandas as pd
import torch
from natsort import natsorted
import glob
import random
import os
import numpy as np
from torchvision import transforms
from PIL import Image


class Nordland(torch.utils.data.Dataset):
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

        self.images = glob.glob(os.path.join(self.data_path, "*/*/*.png"))
        self.seasons = [(f.path, f.name) for f in os.scandir(data_path) if f.is_dir()]
        self.sections = [(f.path, f.name) for f in os.scandir(self.seasons[0][0]) if f.is_dir()]
        self.maxnum = {}
        for path, name in self.sections:
            files = natsorted(glob.glob(os.path.join(path, "*.png")))
            first_filename = os.path.splitext(os.path.basename(files[0]))[0]
            last_filename = os.path.splitext(os.path.basename(files[-1]))[0]
            self.maxnum[name] = [int(first_filename), int(last_filename)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anc_path = self.images[idx]
        pos_path = self.get_positive(anc_path)
        neg_path = self.get_negative(anc_path)
        another_neg_path = self.get_another_negative(anc_path, neg_path) 
        anc = Image.open(anc_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        another_neg = Image.open(another_neg_path).convert('RGB')
        anc = self.transform(anc)
        pos = self.transform(pos)
        neg = self.transform(neg)
        another_neg = self.transform(another_neg)

        return anc, pos, neg, another_neg

    def get_positive(self, path):
        season_src = path.split("/")[-3]
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            season_dst = random.choice(self.seasons)[1]
            number_dst = random.randrange(max(self.maxnum[section_src][0], number_src-self.window), min(number_src+self.window+1, self.maxnum[section_src][1]))
            if season_src != season_dst or number_src != number_dst:
                break

        path = path.replace(season_src, season_dst)
        path = path.replace(str(number_src)+ext, str(number_dst)+ext)

        return path

    def get_negative(self, path):
        season_src = path.split("/")[-3]
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0])
        ext = os.path.splitext(path.split("/")[-1])[1]

        while True:
            season_dst = random.choice(self.seasons)[1]
            section_dst = random.choice(self.sections)[1]
            number_dst = random.randrange(self.maxnum[section_dst][0], self.maxnum[section_dst][1]+1)
            if season_src != season_dst or section_src != section_dst or abs(number_src - number_dst) > self.window:
                break

        path = path.replace(season_src, season_dst)
        path = path.replace(section_src, section_dst)
        path = path.replace(str(number_src)+ext, str(number_dst)+ext)

        return path
    
    def get_another_negative(self, path, first_neg_path):
        season_src = path.split("/")[-3]
        section_src = path.split("/")[-2]
        number_src = int(os.path.splitext(path.split("/")[-1])[0])
        ext = os.path.splitext(path.split("/")[-1])[1]

        # 첫 번째 negative 이미지의 정보 추출
        first_neg_season = first_neg_path.split("/")[-3]
        first_neg_section = first_neg_path.split("/")[-2]
        first_neg_number = int(os.path.splitext(first_neg_path.split("/")[-1])[0])

        while True:
            season_dst = random.choice(self.seasons)[1]
            section_dst = random.choice(self.sections)[1]
            number_dst = random.randrange(self.maxnum[section_dst][0], self.maxnum[section_dst][1] + 1)

            # 첫 번째 negative와 다르고 anchor와도 충분히 다른지 확인
            if ((season_src != season_dst or section_src != section_dst or abs(number_src - number_dst) > self.window) and
                (first_neg_season != season_dst or first_neg_section != section_dst or first_neg_number != number_dst)):
                break

        path = path.replace(season_src, season_dst)
        path = path.replace(section_src, section_dst)
        path = path.replace(str(number_src)+ext, str(number_dst)+ext)

        return path