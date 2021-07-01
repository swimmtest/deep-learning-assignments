import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm
from opts import parser
from dataset import build_word2id, get_train_data, build_word2vec, load_word2vec
from model import TextCNN, LSTM
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config():
    def __init__(self):
        self.batch_size = 128
        self.lr = 1e-3
        self.embedding_dim = 50
        # hidden_dim = TextCNN 80, LSTM 40
        self.hidden_dim = 80
        self.sequence_length = 60 
        self.drop_keep_prob = 0.5    

        # LSTM
        self.layer_size = 2
        self.bidirectional = False
        self.num_direction = (2 if self.bidirectional else 1)
        
        # TextCNN
        self.update_w2v = True               
        self.num_filters = 256           
        self.kernel_size = 3             
        self.pretrained_embed = None 

def train(train_loader, val_loader, model, criterion, optimizer, args, cfg, writer):
    best_acc = 0
    best_model = None

    for epoch in range(args.epoch):
        train_loss = 0
        correct = 0
        total = 0

        model.train()
        print("Epoch{}:".format(epoch+1))
        for i, (x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y = x.to(device), y.to(device)   
            out = model(x)

            # backward  
            optimizer.zero_grad()                                                                
            loss = criterion(out, y)      
            loss.backward() 

            # update weights                          
            optimizer.step() 

            train_loss += loss.data.item()
            _, pred = torch.max(out, 1)

            total += y.size(0)
            correct += (pred == y).sum()

        print('epoch [{}]: train loss is:{:.6f}, train acc is:{:.6f}'
              .format(epoch+1, train_loss / (len(train_loader) * cfg.batch_size), correct / total))
        writer.add_scalar("train_loss", train_loss / (len(train_loader) * cfg.batch_size), epoch+1)
        # val
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for j, (x,y) in tqdm(enumerate(val_loader)):
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.data.item()

                _, pred = torch.max(out, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum()
        
        print('val loss is:{:.6f}, val acc is:{:.6f}'
                .format(val_loss / (len(val_loader) * cfg.batch_size), val_correct / val_total))
        writer.add_scalar("val_loss", val_loss / (len(val_loader) * cfg.batch_size), epoch+1)
        # save best model
        if best_acc < val_correct / val_total:
            best_acc = val_correct / val_total
            best_model = model.state_dict()
            torch.save(best_model, args.best_model)
            print('best acc is {:.6f}, best model is changed'.format(best_acc))

    torch.save(model.state_dict(), args.latest_model)


if __name__ == '__main__':
    cfg = Config()
    args = parser.parse_args()
    writer = SummaryWriter("log")

    print("Load Word2id...")
    word2id, tag2id = build_word2id(args.train_path, args.val_path, args.test_path)
    print("Load Train Data...")
    x_train, y_train, x_val, y_val = get_train_data(args.train_path, args.val_path, word2id, tag2id)

    # TextCNN
    print("Load Pretrained Word2vector...")
    if args.save_w2v:
        cfg.pretrained_embed = load_word2vec(args.save_w2v)
    else: 
        cfg.pretrained_embed = build_word2vec(word2id, args)

    train_data = Data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    val_data = Data.TensorDataset(torch.LongTensor(x_val), torch.LongTensor(y_val))

    train_loader = Data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = Data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    vocab_size = len(word2id)
    tag_size = len(tag2id)

    # model
    model = TextCNN(vocab_size, tag_size, cfg).to(device)
    # model = LSTM(vocab_size, tag_size, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train(train_loader, val_loader, model, criterion, optimizer, args, cfg, writer)
    



