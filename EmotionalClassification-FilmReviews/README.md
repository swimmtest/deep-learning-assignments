# Emotional classification of film reviews

This code is implemented to classify film reviews by PyTorch using TextCNN(pretrained wiki word_vectors) and LSTM, and also use tensorboard to monitor loss.

## Dependency

torch==1.7.1

gensim==4.0.1

tqdm==4.61.1

numpy==1.20.2

## Dataset

* **train**: about 20000 film reviews, pos/neg separately 10000
* **validation**: about 6000 film reviews, pos/neg separately 3000
* **test**: about 360 film reviews, pos/neg separately 180

## Results

|  model  | Accuracy  | Precision |  Recall   |    F1     |
| :-----: | :-------: | :-------: | :-------: | :-------: |
|  LSTM   |   0.846   |   0.874   |   0.813   |   0.842   |
| TextCNN | **0.851** | **0.875** | **0.824** | **0.848** |

## Reference

I mainly refer to [Text-classification](https://github.com/WhatTong/Text-classification) and [Text-CNN](https://github.com/0809zheng/Chinese-movie-comments-sentiment-analysis-pytorch).

