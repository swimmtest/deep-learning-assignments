import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default="dataset/train.txt")
parser.add_argument('--val_path', type=str, default="dataset/validation.txt")
parser.add_argument('--test_path', type=str, default="dataset/test.txt")
parser.add_argument('--best_model', type=str, default="model/best_model.pth")
parser.add_argument('--latest_model', type=str, default="model/latest_model.pth")
parser.add_argument('--w2v_data_path', type=str, default="dataset/wiki_word2vec_50.bin")
parser.add_argument('--save_w2v', type=str, default="dataset/w2v.json")
parser.add_argument('--epoch', type=int, default=20)



    




