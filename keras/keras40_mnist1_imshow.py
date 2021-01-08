# 인공지능계의 hello world라 불리는 mnist를 써보자

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)   #28,28,1 과 28,28 은 같은 거니 흑백이다.
print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000,)

print(x_train[0])                       #이 데이터의 값이 궁금하면
print(y_train[0])                       #라벨을 확인해보자

print(x_train[0].shape)                 #(28, 28)

#이 [0]째의 이미지를 봐보자
plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0]) #컬러를 지정하지 않아도 나오는데 우리가 원하는 방식은 아니다.
plt.show()  #흑백이 데이터로는 0이다. 가장 밝은 것이 255