import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, tag_size, cfg):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        # classes
        self.tag_size = tag_size
        self.embedding_dim = cfg.embedding_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.layer_size
        self.bidirectional = cfg.bidirectional
        self.num_direction = cfg.num_direction
        self.drop_keep_prob = cfg.drop_keep_prob

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, 
                            bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.drop_keep_prob)
        self.fc = nn.Linear(self.hidden_dim * self.num_direction, self.tag_size)

    def forward(self, input):
        input = self.embedding(input).permute(1,0,2)
        out, _ = self.lstm(input)
        out = out[-1, :, :]
        out = out.view(-1, self.hidden_dim * self.num_direction)
        out = self.fc(out)

        return out

class TextCNN(nn.Module):
    def __init__(self, vocab_size, tag_size, cfg):
        super(TextCNN, self).__init__()
        self.update_w2v = cfg.update_w2v
        # classes
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.embedding_dim = cfg.embedding_dim
        self.num_filters = cfg.num_filters
        self.kernel_size = cfg.kernel_size
        self.drop_keep_prob = cfg.drop_keep_prob
        self.pretrained_embed = cfg.pretrained_embed

        # use pretrained word2vector
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_embed))
        self.embedding.weight.requires_grad = self.update_w2v

        self.conv = nn.Conv2d(1, self.num_filters, (self.kernel_size, self.embedding_dim))
        self.dropout = nn.Dropout(self.drop_keep_prob)
        self.fc = nn.Linear(self.num_filters, self.tag_size)

    def forward(self, x):
        x = x.to(torch.int64)         # (batch_size, sequence_length)
        x = self.embedding(x)         # (batch_size, sequence_length, embedding_dim)
        x = x.unsqueeze(1)            # (batch_size, 1, sequence_length, embedding_dim)
        x = F.relu(self.conv(x)).squeeze(3)         # (batch_size, num_filters, sequence_length-2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)   # (batch_size, num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return x