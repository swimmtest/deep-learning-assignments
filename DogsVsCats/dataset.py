import torch.utils.data as data
import os
import random
from PIL import Image


class DogsVSCatsDataset(data.Dataset):      
    def __init__(self, data_dir, mode, split_n=0.9, rng_seed=620, transform=None):  
        self.data_dir = data_dir        
       
        self.split_n = split_n
        self.rng_seed = rng_seed
        self.transform = transform  
        self.data_info = self._get_data_info()

    def __getitem__(self, item):            
        path_img, label = self.data_info[item]
        # 0 ~ 255
        img = Image.open(path_img).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)  

    def _get_data_info(self):               
        img_names = os.listdir(self.data_dir)
        # filter through .jpg
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
        if self.mode in ["train", "val"]:   
            # shuffle
            random.seed(self.rng_seed)
            random.shuffle(img_names)

            # labels
            img_labels = [0 if n.startswith('cat') else 1 for n in img_names]

            # split into train and val dataset
            split_idx = int(len(img_labels) * self.split_n)

            if self.mode == "train":
                img_set = img_names[:split_idx]
                label_set = img_labels[:split_idx]
            elif self.mode == "val":
                img_set = img_names[split_idx:]
                label_set = img_labels[split_idx:]

        elif self.mode == "test":
            # labels
            img_labels = [0 if n.startswith('cat') else 1 for n in img_names]
            img_set = img_names
            label_set = img_labels

        else:
            raise Exception("mode can not recognize, only support (train, val, test)")

        # image abspath
        path_img_set = [os.path.join(self.data_dir, img) for img in img_set]
        data_info = [(p_img, label) for p_img, label in zip(path_img_set, label_set)]

        return data_info
