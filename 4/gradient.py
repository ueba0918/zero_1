#重みパラメータに関する損失関数の損失を抑えるために、微分を利用し勾配が低い方へ点を移動させる必要がある。
import numpy as np


#数値微分の実装
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)

#偏微分(1変数のみ)の実装
def function_2(x):
    return x[0]**2 + x[1]**2

#偏微分(2変数)の実装
#全ての変数の偏微分をベクトルとしたものを勾配という。
#勾配は各地点において低くなる方向を指す。しかし、鞍点や関数の極小値、最小値では勾配が0となってしまう。
def numerical_gradient(f, x):
    h = 1e-4#0.0001
    grad = np.zeros_like(x)#xと同じ形の配列を作成

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val#元の値に戻す

        return grad


#勾配法の実装
# x_n = x_n - η(∂f/∂x_n)
#エータは学習率で1回の学習でどれだけパラメータを更新するかを決める変数
#　fは最適化したい関数、init_xは初期値、lrは学習率、step_numは繰り返す数。
def gradient_decent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
