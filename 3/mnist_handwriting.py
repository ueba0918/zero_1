import sys, os
sys.path.append('C:/Users/uy011/Desktop/deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


#MNIST画像の表示関数
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

#訓練画像、訓練ラベル、テスト画像、テストラベルの読み込み
#flattenは入力画像を1次元配列にするかどうかを設定する。Trueだと784個の要素からなる1次元配列となる。Falseにすると入力画像は1*28*28の3次元配列となる。
#normalizeは入力画像を0.0~1.0の値に正規化するかどうかを設定する。Falseにすると入力画像のピクセルは元の0~255のままになる。
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


img = x_train[0]
label = t_train[0]
print(label)

#flattenがTrueなので、1列で表現されている。これを28*28の形に変形(reshape)する必要がある。
print(img.shape)#784
img = img.reshape(28, 28)
print(img.shape)#28*28

#画像の表示
img_show(img)
