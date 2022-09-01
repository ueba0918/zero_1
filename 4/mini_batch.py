from json import load
import sys, os
sys.path.append('C:/Users/uy011/Desktop/deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)#60000,784の訓練データ
print(t_train.shape)#60000,10のラベル


#この訓練データからランダムに10個抜き出す処理
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)#train_sizeからランダムにbatch_size分だけ抜き出す。
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

