#順伝播グラフは、普通の計算を表し、その逆伝播は局所的な微分(連鎖律により)を表す。
#単純なレイヤの実装

#乗算レイヤの実装
class MullLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    #上流から伝わってきた微分に対して、順伝播のひっくり返した値を乗算して下流に流す。
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

#加算レイヤの実装
class AddLayer:
    #初期化は必要ないのでpass
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
