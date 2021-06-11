import torch.utils.data as data
import os
from PIL import Image


class DogsVSCatsDataset(data.Dataset):      
    def __init__(self, data_dir, mode, split_n=0.9, rng_seed=620, transform=None):  
        self.data_dir = data_dir        
        self.mode = mode
        self.split_n = split_n
        self.rng_seed = rng_seed
        self.transform = transform 
        self.img_names = os.listdir(self.data_dir)

    def __getitem__(self, item):        
        img = Image.open(os.path.join(self.data_dir, self.img_names[item])).convert('RGB')
        if self.transform:
            img = self.transform(img)  

        if self.mode in ["train", "val"]:        
            label = 0 if self.img_names[item].startswith("cat") else 1
            return img, label
        else:
            return img, self.img_names[item]

    def __len__(self):
        return len(self.img_names)