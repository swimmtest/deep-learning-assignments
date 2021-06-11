import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_latest_model', type=str, default='model/latest_model.pth')
parser.add_argument('--train_data', type=str, default='data/train')
parser.add_argument('--test_data', type=str, default='data/test')
parser.add_argument('--save_info', type=str, default='data/tensorboard')



