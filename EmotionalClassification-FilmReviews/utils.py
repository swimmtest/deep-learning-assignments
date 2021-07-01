import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(pred, label):
    zeros = Variable(torch.zeros(label.size()).type(torch.LongTensor)).to(device)
    ones = Variable(torch.ones(label.size()).type(torch.LongTensor)).to(device)
    TP = ((pred == ones) & (label == ones)).sum()
    TN = ((pred == zeros) & (label == zeros)).sum()
    FP = ((pred == ones) & (label == zeros)).sum()
    FN = ((pred == zeros) & (label == ones)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, F1, accuracy

