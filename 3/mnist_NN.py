from copyreg import pickle
import sys, os
sys.path.append('C:/Users/uy011/Desktop/deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import active_function as ac
import pickle


#ニューラルネットワークを実装する。
#入力層は784個で、出力層は10個(0~9の数字分類なので)

#隠れ層は2つで、一つ目の隠れ層が50個、2つ目が100個のニューロンを持つものとする。

#データの入手
#normalizeはTrueにすると正規化（0~255を255で割って0~1の間に変換）する。
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test,t_test

#ファイルのオープンと読み込み
#pickleファイルのsample_weight.pklに保存された学習済みの重みパラメータを読み込む。
#pickleファイルとは、プログラム実行中のオブジェクトをファイルとして保存する機能。
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

#NNの作成
def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = ac.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = ac.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    #確率値に変換
    y = ac.softmax(a3)

    return y

#xとtに訓練データを代入
x, t = get_data()

#networkに重みパラメータを代入
network = init_network()

#バッチの数
batch_size = 100

accuracy_cnt = 0

#xに格納された画像データを1枚ずつ取り出し、predictによって分類を行う。
#predictは各ラベルの確率が配列として出力される。0の結果が0.1のような形で出力される。
#そのうち一番高いモノを予測結果とする。
for i in range(0, len(x), batch_size):#0からbatch_sizeで指定された値だけ増加した数値のrange
    x_batch = x[i:i+batch_size]#iからi+batch_n番目までのデータを取り出す。先頭から100枚ずつバッチとして取り出す。
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
