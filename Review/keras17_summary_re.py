import numpy as np
import tensorflow as tf

#모델 구성하고 써머리 보기
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#시퀀셜 모델
model = Sequential()
model.add(Dense(3, input_dim=1, activation='relu'))
model.add(Dense(6))
model.add(Dense(9))
model.add(Dense(12))
print('sequentail model')
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 24
_________________________________________________________________
dense_2 (Dense)              (None, 9)                 63
_________________________________________________________________
dense_3 (Dense)              (None, 12)                120
=================================================================
Total params: 213
Trainable params: 213
Non-trainable params: 0
_________________________________________________________________
'''

#함수형 모델
inputs = Input(shape=(1,))
dense1 = Dense(5, activation='relu')(inputs)
dense1 = Dense(10)(dense1)
outputs = Dense(15)(dense1)
model = Model(inputs = inputs, outputs = outputs)
print('fuctional model')
model.summary()

'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 10)                60
_________________________________________________________________
dense_5 (Dense)              (None, 15)                165
=================================================================
Total params: 235
Trainable params: 235
Non-trainable params: 0

_________________________________________________________________
'''