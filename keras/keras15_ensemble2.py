# 실습 y2를 없애서 다(2):1로 바꾸자. 분기를 안한다는 힌트

import numpy as np

#1. 데이터 제공
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411,511), range(100,200)])
y1 = np.array([range(711, 811), range(1,101), range(201,301)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.8, shuffle = False)
#이렇게 세개도 가능

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)

# 모델 병합 / concatenate = 사슬처럼 엮다
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1) 

#모델 분기 #이번에는 나누는게 아니니까 아웃풋을 한 모델만 두어도 된다.
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)

#4. 평가, 예측
loss, metrics = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)

print('model.metrics_names: ', model.metrics_names)
print('loss, metrics: ', loss, metrics)

y1_predict= model.predict([x1_test, x2_test])

print('================================')
print('y1_predict :\n', y1_predict)
print('================================')

#y1_test, y2_test와 비교하자 RMSE랑 R2로
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE = RMSE(y1_test, y1_predict)
print('RMSE: ', RMSE)
# RMSE정의에 구지 y1이라고 안 해줘도 괜찮다. r2도 마찬가지

from sklearn.metrics import r2_score

R2= r2_score(y1_test, y1_predict)
print('R2: ', R2)
