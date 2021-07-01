import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm
from opts import parser
from dataset import build_word2id, get_test_data, load_word2vec
from model import TextCNN, LSTM
from train import Config
from utils import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_loader, model, criterion, cfg):
    model.eval()
    
    correct = 0
    total = 0
    test_loss = 0
    all_pred = []
    all_y = []
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_loader)):
            x, y = x.to(device), y.to(device)
            out = model(x)

            loss = criterion(out, y)
            test_loss += loss.data.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum()
            total += y.size(0)

            all_pred.append(pred)
            all_y.append(y)

    precision, recall, F1, accuracy = evaluate(torch.cat(all_pred, dim=0), torch.cat(all_y, dim=0))
    print('test loss is:{:.6f}, test acc is:{:.6f}'
              .format(test_loss / (len(test_loader) * cfg.batch_size), correct / total))
    print('test precision is:{:.6f}, test recall is:{:.6f}, test F1 is:{:.6f}, test accuracy is:{:.6f}'
              .format(precision, recall, F1, accuracy))
            


if __name__ == '__main__':
    cfg = Config()
    args = parser.parse_args()

    print("Load Word2id...")
    word2id, tag2id = build_word2id(args.train_path, args.val_path, args.test_path)
    print("Load Test Data...")
    x_test, y_test = get_test_data(args.test_path, word2id, tag2id)

    # TextCNN
    print("Load Pretrained Word2vector...")
    if args.save_w2v:
        cfg.pretrained_embed = load_word2vec(args.save_w2v)

    test_data = Data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    test_loader = Data.DataLoader(test_data, batch_size=cfg.batch_size)

    vocab_size = len(word2id)
    tag_size = len(tag2id)

    # model
    model = TextCNN(vocab_size, tag_size, cfg)
    # model = LSTM(vocab_size, tag_size, cfg)
    # load best model
    model.load_state_dict(torch.load(args.best_model))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test(test_loader, model, criterion, cfg)