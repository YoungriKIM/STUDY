# 커스텀 로스를 해보자 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


def custom_mse(y_true, y_pred):                             # y_true, y_pred 의 이름을 바꿔도 알아서 y원래값과 y예측값으로 들어간다.
    return tf.math.reduce_mean(tf.square(y_true-y_pred))    # 이 식이 원래 mse의 식  #square = 제곱한 것


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')   # 이렇게 실수형으로 바꾸든
# x = np.array([1.,2.,3.,4.,5.,6.,7.,8.]).astype('float32')   # 이렇게 실수형으로 바꾸든
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')
print(x.shape)    #(8, )  스칼라가 8개

#2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=custom_mse, optimizer='adam')

model.fit(x, y, epochs=50, batch_size=1)

loss = model.evaluate(x,y)
print(loss)

# 0.00034028213121928275
