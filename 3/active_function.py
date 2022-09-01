#活性化関数とは、閾値を境にして出力が切り替わる関数で、入力信号の総和を出力信号に変換する。

import numpy as np
import matplotlib.pylab as plt
#ステップ関数：入力が0を越えたら1を出力し、それ以外は0を出力する関数
#配列でも対応できるようにする。
def step_function(x):

    #boolean型で配列の中身をTrueかFalseで返す。
    y = x > 0

    #yの要素の型をintに変換し、astypeで0か1を出力する。
    return y.astype(np.int)

#ステップ関数の表示
#x = np.arange(-5.0, 5.0, 0.1)
#y = step_function(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)#y軸の範囲を指定
#plt.show()


#シグモイド関数の実装
#シグモイド関数は、0~1の滑らかな曲線となる。
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#シグモイド関数の表示
#x = np.arange(-5.0, 5.0, 0.1)
#y = sigmoid(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()


#ReLU関数の実装
#ReLU関数は、入力が0以下ならば0、0より大きければその値を出力する。
def relu(x):
    return np.maximum(0,x)

#入力値をそのまま返す恒等関数
def identity_function(x):
    return x


#ソフトマックス関数
#0~1の実数に変換する関数で、確率として解釈することができる。
#よって、出力層の活性化関数として利用することが多いが、計算量が多くなってしまうため省略することがある。
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)#オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
