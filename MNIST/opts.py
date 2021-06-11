import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--save_latest_model', type=str, default='model/latest_model.pth')
parser.add_argument('--save_latest_optimizer', type=str, default='model/latest_optimizer.pth')
