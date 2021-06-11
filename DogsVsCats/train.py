import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import os
from dataset import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader
from network import Net
from opts import parser
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWritter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config():
    def __init__(self):
        self.batch_size = 64                     
        self.lr = 0.005                         
        self.n_epoch = 60
        # self.workers = 1
        self.n_classes = 2
        self.lr_decay_step = 15
        self.log_interval = 1
        self.val_interval = 1
        self.plot = True

def train(train_loader, val_loader, model, optimizer, criterion, scheduler, n_epoch, log_interval, val_interval, save_latest_model, plot=False):                                
    train_curve = list()
    val_curve = list()

    for epoch in range(1, n_epoch+1):
        correct = 0
        loss_mean = 0
        total = 0

        model.train()
        for i, (img,label) in enumerate(train_loader):
            # forward
            img, label = img.to(device), label.to(device)           
            out = model(img)   
            
            # backward                                                 
            optimizer.zero_grad()                      
            loss = criterion(out, label)      
            loss.backward()   

            # update weights                          
            optimizer.step() 

            # static class results
            _, pred = torch.max(out.data, 1)  
            total += label.size(0)
            correct += (pred == label).squeeze().cpu().sum().numpy()                        

            # print train information
            loss_mean += loss.item()

            # writer.add_scalar('Train/Loss', loss.item(), (i+1)*epoch)
            # writer.flush()

            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                # mean loss on mini-batch every log_interval
                loss_mean = loss_mean / log_interval
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%}".format(                   
                    epoch, n_epoch, i+1, len(train_loader), loss_mean, correct/total))
                loss_mean = 0

        # update learning rate
        scheduler.step()

        # validate the model
        if epoch % val_interval == 0:
            correct_val = 0
            total_val = 0
            loss_val = 0

            model.eval()

            with torch.no_grad():
                for j, (img,label) in enumerate(val_loader):
                    img, label = img.to(device), label.to(device)
                    bs, n_crops, c, h, w = img.size()

                    # output by training model
                    out = model(img.view(-1, c, h, w))   
                    out_avg = out.view(bs, n_crops, -1).mean(1)

                    # loss
                    loss = criterion(out_avg, label)

                    _, pred = torch.max(out_avg.data, 1)
                    total_val += label.size(0)
                    correct_val += (pred == label).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                # mean loss on mini-batch
                loss_val_mean = loss_val / len(val_loader)
                val_curve.append(loss_val_mean)
                print("Valid:\t  Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, n_epoch, j+1, len(val_loader), loss_val_mean, correct_val/total_val))

            model.train()

    # save latest model
    torch.save(model.state_dict(), save_latest_model)

    # plot loss
    if plot == True:
        # train
        train_x = range(len(train_curve))
        train_y = train_curve
        # val
        train_iters = len(train_loader)
        val_x = np.arange(1, len(val_curve)+1) * train_iters * val_interval
        val_y = val_curve

        # plot configure
        plt.plot(train_x, train_y, label='train')
        plt.plot(val_x, val_y, label='val')

        plt.legend(loc='upper right')
        plt.xlabel('Iteration')
        plt.ylabel = ('Loss')
        plt.show()


def main():
    # configure
    cfg = Config()
    args = parser.parse_args()
    # writer = SummaryWritter(args.save_info)

    # data preprocessing
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # train transform
    train_transform = transforms.Compose([
    transforms.Resize((256)),                         
    transforms.CenterCrop(256),        
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),   
    transforms.Normalize(norm_mean, norm_std),
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    # val transform
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])


    # build train and val dataset
    train_data = DVCD(args.train_data, 'train', transform=train_transform)                                                               
    val_data = DVCD(args.train_data, 'val', transform=val_transform)                                                           

    # build data_loader
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size)
    print('Train dataset loaded! length of train set is {0}'.format(len(train_data)))
    print('Val dataset loaded! length of val set is {0}'.format(len(val_data)))

    # model
    model = Net(cfg.n_classes)

    # model = get_model(cfg.n_classes, pretrained=False) 
    model = model.to(device) 

    # network configure
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9) 
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=0.1)

    # judge model dir exists
    dir_model = os.path.dirname(os.path.abspath(args.save_latest_model))
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    # train
    train(train_loader, val_loader, model, optimizer, criterion, scheduler, cfg.n_epoch, cfg.log_interval, cfg.val_interval, args.save_latest_model, cfg.plot)
    

if __name__ == '__main__':
    main()









