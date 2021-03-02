import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

# 데이터/ 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train[0], x_train[1]*x_train[2]).astype('float32')/255.
x_test = x_test.reshape(x_test[0], x_test[1]*x_test[2]).astype('float32')/255.

# 모델 구성
def build_model(drop=0.5, optimazer='adam'):
    