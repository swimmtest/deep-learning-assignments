# DogsVsCats

This code is implemented by PyTorch to classify dogs and cats on kaggle competition, to help learn about deep-learning better.

I use train-data about 2160, val 240, test 600. Test Acc achieves 76.5%. 

## Usage

1. **download dataset**
2. **train**:  `python train.py`
3. **test**:   `python test.py`

## More

Because of the largest dataset, and you don't have GPU. I recommend you run this code in [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb):

1. **set GPU**: Modify->notebook setting->GPU

2. **new .ipynb**:

   ```
   # load google cloud hard disk
   from google.colab import drive
   drive.mount("/content/drive")
   
   # cd to corresponding file
   cd /content/drive/MyDrive/DogsVsCats
   
   # train
   !python train.py
   ```