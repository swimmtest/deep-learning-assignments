import os
import torch
import torch.nn as nn
from dataset import get_data
from tqdm import tqdm
from torchnet import meter
from model import PoetryModel
from opts import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(**kwargs):
    print("start training")
    print("input parameter list: " + str(kwargs.items()))
    cfg = Config()

    for para, value in kwargs.items(): setattr(cfg, para, value)

    print("The model use device: " + str(device))

    print("now start get data, word2ix, ix2word")
    train_data, word2ix, ix2word = get_data(cfg)
    print("get dataset finish!")
    print("dataset content:\n{}".format(train_data))
    print("word2ix size: {}".format(len(word2ix)))
    print("ix2word size: {}".format(len(ix2word)))

    # numpy array to tensor
    train_data = torch.from_numpy(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    print("finish train data loader")

    # define model
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if cfg.pre_train_model_path:
        model.load_state_dict(torch.load(cfg.pre_train_model_path))
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    if not os.path.exists(cfg.model_prefix): os.mkdir(cfg.model_prefix)
    for epoch in range(cfg.epoch):
        print("eopch: {}/{} training".format(epoch+1, cfg.epoch))
        for batch_idx, batch_data in tqdm(enumerate(train_loader), desc="train process"):
            batch_data = batch_data.long().transpose(1, 0).contiguous().to(device)
            optimizer.zero_grad()
            input, target = batch_data[:-1, :], batch_data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

        torch.save(model.state_dict(), os.path.join(cfg.model_prefix, "{}_{}.pth".format(cfg.class_limit, epoch+1)))
        print("save model: ", os.path.join(cfg.model_prefix, "{}_{}.pth".format(cfg.class_limit, epoch+1)))


if __name__ == '__main__':
    import fire
    fire.Fire()