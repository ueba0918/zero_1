#損失関数の実装
#損失関数は、訓練データから最適な重みパラメータの値を自動で獲得する。
#指標を認識精度にしてはいけない理由として、パラメータの微分が0となってしまう問題があるから。
#ニューラルネットワークの出力y_kは確率値、訓練データt_kはone-hot表現のラベルとする。

import numpy as np


#2乗和誤差
# 1/2 sum(y_k - t_k)^2
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


#交差エントロピー誤差
# -sum(t_k log y_k)
#バッジ対応版
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
