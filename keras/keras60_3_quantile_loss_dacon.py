# 커스텀 로스를 해보자 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

#===========================================================

# mse 만들기
def custom_mse(y_true, y_pred):                             # y_true, y_pred 의 이름을 바꿔도 알아서 y원래값과 y예측값으로 들어간다.
    return tf.math.reduce_mean(tf.square(y_true-y_pred))    # 이 식이 원래 mse의 식  #square = 제곱한 것


# 퀀타일 로스1를 지정해주자
def quantile_loss(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)          # constant = 상수라는 뜻
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

# 데이콘 베이스라인에서 가져온 퀀타일 로스
def quantile_loss_dacon(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#===========================================================

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

# model.compile(loss=quantile_loss, optimizer='adam')
model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles[0], y_true, y_pred), optimizer='adam')     # 데이콘에서 가져온 것 

# quantiles[0] = 0.1 이다.
# lambda y_true, y_pred: quantile_loss(q, y_true, y_pred)
#        -------------   --------------------------------
#        이거를           여기에 넣겠다.


model.fit(x, y, epochs=50, batch_size=1)

loss = model.evaluate(x,y)
print(loss)

# mse 만들어준거
# 0.00034028213121928275

# 퀀타일1 만든거
# 0.008313745260238647

# 퀀타일 [0]째 거
# 0.0066979192197322845
