#R2 : 0.9 나올때까지 실습
#시퀀셜도 가능하니까 해보기

import numpy as np

from sklearn.datasets import load_boston #사이킷런에서 제공하는 데이터셋 중 보스턴 집값을 이용하겠다.

dataset = load_boston()
x = dataset.data
y = dataset.target

# print(x.shape) #(506, 13)
# print(y.shape) #(506,) #그럼 이걸 mlp로 칼럼이 여러개인 걸로 모델을 짤 수 있겠군
# print('=================================')
# print(x[:5]) #한 셋에 13개가 들어있음
# # [6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00 6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02 4.9800e+00]
# # 그런데 이 숫자가 내가 본 자료랑은 좀 다르다. 왜냐면 numpy는 단순연산,소수점 연산에 특화되어있고, 데이터가 너무 크면 연산이 힘들어 1과0사이로 데이터전처리가 되어있는 것이다.
# print(y[:10])
# print('=================================')
# print(np.max(x), np.min(x)) #최대값과 최소값 > 711.0 0.0 > 데이터 전처리 방법이 한 가지가 아니라서 최대값이 711. 으로 나온 것
# print('=================================')
# print(dataset.feature_names) #열의 이름
# print(dataset.DESCR) #특성(=열) describe묘사하다/ 특성의 설명이 나옴

#데이터 전처리 (MinMax) 
# x = x /711.
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
print(np.max(x[0])) #x의 0번째 칼럼의 max값임

from sklearn.preprocessing import MinMaxScaler #preprocessing = 전처리
scaler = MinMaxScaler() #민맥스캘러를 스캘러라고 부르겠삼
scaler.fit(x) #민맥스캘러를 해라 x에
x = scaler.transform(x) #모든 x의 칼럼에 711을 나눈 것이 아니라 각 칼럼에 맞는 최대,최소를 구해서 해주는 것이다. 위의 x = x/711은 칼럼별이 아니다.

print(np.max(x), np.min(x)) #1.0 0.0
#이러면 내가 데이터의 최대 최소를 몰라도 전처리를 할 수 있다.
print(np.max(x[0]))


#트레인이랑 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#모델 짜기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(13,)) #input_dim=13 이랑 input_shape=(13,) 과도 같다.
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(inputs)
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(dense1)
dense1 = Dense(56, activation='relu')(dense1)
outputs = Dense(1, activation='relu')(dense1)

model = Model(inputs= inputs, outputs = outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=2, validation_split=0.2, verbose=1)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss, mae : ', loss, mae)

y_predict1 = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict1))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict1)
print('R2: ', R2)

#전처리 전
# loss, mae :  19.918365478515625 3.237553119659424
# RMSE:  4.462999550425985
# R2:  0.7616934074949592

#전처리 후 x = x/711.
# loss, mae :  11.487313270568848 2.5379178524017334
# RMSE:  3.3892934685148166
# R2:  0.8625639325151193

#MinMaxscaler 적용 후
# loss, mae :  10.720622062683105 2.2357163429260254
# RMSE:  3.2742366375542535
# R2:  0.871736673764582