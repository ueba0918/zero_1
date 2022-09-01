#確率的勾配降下法(stochastic gradient descent)は傾斜がきつい方向へ進んでいこうというもの。
#SGDは関数によっては、本来の最小値ではない方向を指してしまう場合がある。
#最適な重みを探す最適化手法について
import numpy as np

#SGDの実装
#W = W - η * ∂L/∂W
#lrはlearing_rate
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

#Momentumの実装
#v = αv - η * ∂L/∂W
#W = W + v
#Wは更新する重みパラメータ、ηは学習率、∂L/∂WはWに関する損失関数の勾配、vは速度、αvは物体が何も力を受けないときに徐々に減速するための役割=摩擦や空気抵抗に対応する。
#動き方としては物体が勾配によって力を受け、その力によって物体の加速度が加算されるという物理法則を利用し、ボールが地面の傾斜を転がるように動くイメージ。
#SGDと比べ、ジグザクの動きを軽減できる。
class momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

#AdaGradの実装
#h = h + ∂L/∂W ⦿ ∂L/∂W
#W = W - η * 1/√h * ∂L/∂W
#パラメータの要素ごとに適応的に学習係数を調整しながら学習を行う手法
#⦿は行列の要素ごとの掛け算
#hに勾配の2乗和を保持していき、パラメータ更新の際に1/√hを乗算することで、学習のスケールを調整する。
#つまり、大きく動いたパラメータは学習係数が少ないことを意味する。
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

#Adamの実装
#AdaGradとMomentumを融合したもの
