#活性化関数レイヤの実装
import numpy as np
import sys, os
sys.path.append('C:/Users/uy011/Desktop/deep-learning-from-scratch-master')
from common.functions import softmax, cross_entropy_error

#ReLUレイヤの実装
#ReLUはy = 0 (x≦0),y = x (x>0)
#maskはインスタンス変数で、True/FalseからなるNumpy配列で、順伝播の入力であるxの要素でx以下は0、それ以外をFalseとして保持する。
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

#sigmoidレイヤの実装
# y = 1 / (1 + exp(-x) )
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx



#Affineレイヤの実装
#NNの順伝播で行う行列積はアフィン変換と呼ばれる。
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


#Softmax-with-Lossレイヤの実装
#出力層となる確率へ変換するソフトマックス関数
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None#損失
        self.y = None#softmaxの出力
        self.t = None#教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
