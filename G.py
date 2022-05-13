from pathlib import Path


DATA_PATH = Path('/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data')

TRAIN_DATA_PATH = DATA_PATH.joinpath('train.npy')

VAL_DATA_PATH = DATA_PATH.joinpath('val.npy')

TEST_DATA_PATH = DATA_PATH.joinpath('test.npy')

OUT_DIR = Path('/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/weight')