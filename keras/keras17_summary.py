import numpy as np
import tensorflow as tf

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(10))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
'''
# model이 시퀀셜이라고 뜨고 함수형같은 경우는 functional 으로 뜬다
# 아웃풋셰잎 부분의 뒤쪽은 노드의 갯수이다. param를 파라미더 즉 연산되는 횟수인데, 내가 지정한 노드 '+b' (y=wx+b중에서b)를 하면 노드의 수가 하나씩 더 늘어난다.
# 그렇게 계산하면 param이 이해가 됨

#실습2+과제: ensemble1,2,3,4에 대해서 서머리를 계산하고 이해한 것을 과제로 제출 할 것
#layer를 만들 때 'name' 이란 파라미터에 대해 확인하고 설명할 것 또 왜 반드시 써야할 때는 언제인지: 모델 구성할 때 넣는 것.
#힌트는 머신이 이해할 때 이름이 충돌하지 않게 하려는 이유가 크다.