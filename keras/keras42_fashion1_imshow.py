# datasets = fashion_mnist
# - dataset어떻게 구성되어 있는지 확인

import numpy as np

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# (60000, 28, 28)
# (10000, 28, 28)
# (60000,)
# (10000,)

import  matplotlib.pyplot as plt

# print(x_train[0])
# print(y_train[0])

plt.imshow(x_train[1], 'gray')
plt.show()
