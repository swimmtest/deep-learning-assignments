# Auto-writingPoem

This code is implemented by PyTorch using two-LSTM-network to create Chinese poetry automaticlly including tang and song poetry(also acrostic)

## Dependency

* python 3.8.3 

* torch 1.7.1

* torchnet 0.0.5.1

## Dataset

Dataset from [Chinese-poetry](https://github.com/chinese-poetry/chinese-poetry/tree/master/json). I have uploaded the preprocessing training dataset file: train_song_pickle.npz(255000) and train_tang_pickle.npz(58000).

## Usage

**tang:**

1. train: `python3 train.py train --pickle_file_path "data/train_tang_pickle.npz" --class_limit poet.tang  --model_prefix "checkpoints/tang_json" `

2. test: `python3 test.py generate --pre_train_model_path checkpoints/tang_json/poet.tang_20.pth --prefix_words "閑云潭影日悠悠。" --start_words "湖光秋月兩相和"`

**song:**

1. train: `python3 train.py train --pickle_file_path "data/train_song_pickle.npz" --class_limit poet.song  --model_prefix "checkpoints/song_json" `

2. test: `python3 test.py generate --pre_train_model_path checkpoints/song_json/poet.song_20.pth --prefix_words "閑云潭影日悠悠。" --start_words "湖光秋月兩相和" --pickle_file_path "data/train_song_pickle.npz" `

## Reference

I mainly refer to [LSTM_POET](https://github.com/Niutranser-Li/LSTM_POET).