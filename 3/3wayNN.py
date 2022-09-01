#3層ニューラルネットワークの実装

import numpy as np
import active_function as ac

#重みとバイアスの初期化
def init_network():
    #辞書に入れる
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network,x):
    #辞書からニューロンへ代入していく
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    #第1層の計算 a1 = w_11*x1 + w_12*x2 + b1
    a1 = np.dot(x, W1) + b1
    #活性化関数はシグモイド関数を利用する
    z1 = ac.sigmoid(a1)

    #第2層の計算
    a2 = np.dot(z1, W2) + b2
    z2 = ac.sigmoid(a2)

    #第3層の計算
    a3 = np.dot(z2, W3) + b3

    #最後の活性化関数：そのまま返す関数
    y = ac.identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
