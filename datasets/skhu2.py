import os
import glob
import random
import torch
from natsort import natsorted
from torchvision import transforms
from PIL import Image

class SKHU2(torch.utils.data.Dataset):
    def __init__(self, config, data_path):
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

        self.images = glob.glob(os.path.join(self.data_path, "*/*/positive/*.jpg"))
        self.buildings = [(f.path, f.name) for f in os.scandir(data_path) if f.is_dir()]
        self.sections = [(f.path, f.name) for f in os.scandir(self.buildings[0][0]) if f.is_dir()]
        self.labels = [(f.path, f.name) for f in os.scandir(self.sections[0][0]) if f.is_dir()]
        self.maxnum = {}

        for building in os.scandir(data_path):
            if building.is_dir():
                building_name = building.name
                self.maxnum[building_name] = {}
                for section in os.scandir(building.path):
                    if section.is_dir():
                        section_name = section.name
                        self.maxnum[building_name][section_name] = {}

                        positive_folder = os.path.join(section.path, 'positive')
                        pos_images = glob.glob(os.path.join(positive_folder, '*.jpg'))
                        self.maxnum[building_name][section_name]['positive'] = len(pos_images) - 1

                        negative_folder = os.path.join(section.path, 'negative')
                        neg_images = glob.glob(os.path.join(negative_folder, '*.jpg'))
                        self.maxnum[building_name][section_name]['negative'] = len(neg_images) - 1

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
        parts = path.split(os.sep)
        building_src, section_src = parts[-4], parts[-3]
        
        num_images = self.maxnum[building_src][section_src]['positive']
        anchor_num = int(os.path.splitext(parts[-1])[0][6:])

        while True:
            pos_num = random.randint(0, num_images-1)
            if pos_num != anchor_num:
                break

        pos_path = path.replace(str(anchor_num).zfill(4), str(pos_num).zfill(4))

        return pos_path

    def get_negative(self, path):
        parts = path.split(os.sep)
        building_src, section_src = parts[-4], parts[-3]
        
        num_images = self.maxnum[building_src][section_src]['negative']
        neg_num = random.randint(0, num_images-1)

        neg_path = path.replace('positive', 'negative')
        neg_path = neg_path.replace(os.path.splitext(parts[-1])[0][6:], str(neg_num).zfill(4))

        return neg_path