import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from model import Net
from dataset import get_train_data
from opts import parser

class Config():
    def __init__(self):
        self.n_epochs = 5
        self.batch_size_train = 64
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

def train(train_loader, net, optimizer, epoch, log_interval):
    train_losses = []
    train_counter = []
    sum = 0
    correct = 0
    net.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = net(data)
        # prediction
        pred = output.data.max(1, keepdims=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        sum += data.size(0)

        # loss
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100. * correct / sum))

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def main():
    cfg = Config()
    args = parser.parse_args()

    # build data_loader
    train_loader = get_train_data(cfg.batch_size_train)

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    for epoch in range(1, cfg.n_epochs + 1):
        # judge model dir exists
        dir_model = os.path.dirname(os.path.abspath(args.save_latest_model))
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        # train 
        train(train_loader, net, optimizer, epoch, cfg.log_interval)
        
        # save model parameters
        torch.save(net.state_dict(), args.save_latest_model)
        # torch.save(optimizer.state_dict(), args.save_latest_optimizer)

if __name__ == '__main__':
    torch.manual_seed(0)
    main()
    