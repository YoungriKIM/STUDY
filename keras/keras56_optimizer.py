# optimizer 튜닝을 해보자

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
# 옵티마이저는 경사하강법에 기본을 두고 있다. 옵티마이저는 로스를 최적화 하는 것

# optimizer = Adam(lr=0.1)       #0.00180862657725811
# optimizer = Adam(lr=0.01)      #1.6200373833229198e-13
# optimizer = Adam(lr=0.001)     #5.570655156779403e-13
# optimizer = Adam(lr=0.0001)    #1.6631076505291276e-05           러닝레이트가 너무 커도 안 좋고, 작아도 안 좋다.

# optimizer = Adadelta(lr=0.1)     #0.3358626961708069
# optimizer = Adadelta(lr=0.01)    #9.498313011135906e-05
# optimizer = Adadelta(lr=0.001)   #6.624812126159668
# optimizer = Adadelta(lr=0.0001)  #25.56650733947754

# optimizer = Adamax(lr=0.1)       #0.0006634177989326417
# optimizer = Adamax(lr=0.01)      #3.798561291901148e-12
# optimizer = Adamax(lr=0.001)     #8.313350008393172e-13
# optimizer = Adamax(lr=0.0001)    #0.0024631009437143803

# optimizer = Adagrad(lr=0.1)         #0.6519089937210083
# optimizer = Adagrad(lr=0.01)        #6.731725932240806e-08
# optimizer = Adagrad(lr=0.001)       #0.00026779010659083724
# optimizer = Adagrad(lr=0.001)       #1.5576008081552573e-05

# optimizer = RMSprop(lr=0.1)        #73305344.0
# optimizer = RMSprop(lr=0.01)       #0.17327626049518585
# optimizer = RMSprop(lr=0.001)      #0.09527559578418732
# optimizer = RMSprop(lr=0.0001)     #0.0025082980282604694

# optimizer = SGD(lr=0.1)            #nan       > 너무 커서 튕겨져 나간 것이다.
# optimizer = SGD(lr=0.01)           #nan          
# optimizer = SGD(lr=0.001)          #6.232082341739442e-06
# optimizer = SGD(lr=0.0001)         #0.001723085530102253

# optimizer = Nadam(lr=0.1)          #572.348876953125
# optimizer = Nadam(lr=0.01)         #2.5330849614049744e-13
# optimizer = Nadam(lr=0.001)        #3.0377048005902907e-06
optimizer = Nadam(lr=0.0001)       #1.611968400538899e-05


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print('loss: ', loss, '결과물: ', y_pred)

