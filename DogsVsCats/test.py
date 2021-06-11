import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from network import Net
from dataset import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader
# from network import Net
from opts import parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config():
    def __init__(self):
        self.batch_size = 128                   

def test(test_loader, model, criterion):
    model.eval() 
    
    correct = 0
    loss_mean = 0
    total = 0 
    # test_curve = list()                                                                          

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)

            out = model(img)

            # loss
            loss = criterion(out, label)

            # prediction
            _, pred = torch.max(out.data, 1)

            total += label.size(0)
            correct += (pred == label).squeeze().cpu().sum().numpy()

            loss_mean += loss.item()
        # mean loss on mini-batch
        loss_mean = loss_mean / len(test_loader)
        print("Testing: Loss: {:.4f} Acc: {:.2%}".format(                   
                loss_mean, correct/total))

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

    # model
    model = Net()
    model.load_state_dict(torch.load(args.save_latest_model)) 
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # test
    test(test_loader, model, criterion)


if __name__ == '__main__':
    main()

