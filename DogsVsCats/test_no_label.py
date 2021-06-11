import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import os
from utils import imgDisplay
from network import Net
from dataset_test_no_label import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader
# from network import Net
from opts import parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config():
    def __init__(self):
        self.batch_size = 128                   

def test(test_loader, model, criterion):
    model.eval() 

    id_list = []
    pred_list = []
    # test_curve = list()                                                                          

    with torch.no_grad():
        for img, name in test_loader:
            img = img.to(device)

            out = model(img)

            # prediction
            pred = torch.argmax(out, 1)

            # save pred to .csv
            id_list += [n[:-4] for n in name]
            pred_list += [p.item() for p in pred]
    
    submission = pd.DataFrame({"id":id_list, "label":pred_list})
    submission.to_csv('preds_alexnet.csv', index=False)


def main():
    # configure
    cfg = Config()
    args = parser.parse_args()

    # data preprocessing
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    if not os.path.exists(args.test_data):
        raise Exception("test data path is invalid")

    # test dataset
    test_data = DVCD(args.test_data, 'test', transform=transform)                                                          
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size)
    print('Testing dataset loaded! length of test set is {0}'.format(len(test_data)))

    # images display
    imgDisplay(test_loader)

    # model
    model = Net()
    model.load_state_dict(torch.load(args.save_latest_model)) 
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # test
    test(test_loader, model, criterion)


if __name__ == '__main__':
    main()

