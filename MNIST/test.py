import torch
import torch.nn.functional as F
import numpy as np
from dataset import get_test_data
from model import Net
from opts import parser

class Config():
    def __init__(self):
        self.batch_size_test = 1000

def test(test_loader, net):
    test_loss = 0
    correct = 0
    sum = 0
    test_losses = []
    net.eval()

    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            pred = output.data.max(1, keepdims=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            sum += data.size(0)

            test_loss += F.nll_loss(output, target, size_average=False).item()
            
            test_loss /= data.shape[0]
            test_losses.append(test_loss)

            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                np.mean(test_losses), correct, sum,
                100. * correct / sum)) 

def main():
    cfg = Config()
    args = parser.parse_args()

    # build data_loader
    test_loader = get_test_data(cfg.batch_size_test)

    net = Net()
    # load model
    net.load_state_dict(torch.load(args.save_latest_model))
    test(test_loader, net)

if __name__ == '__main__':
    main()