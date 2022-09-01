import sys, os
sys.path.append('C:/Users/uy011/Desktop/deep-learning-from-scratch-master')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simple_Net:
    def ___init__(self):
        self.W = np.random.randn(2,3)#ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


