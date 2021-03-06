# m46_5 가져와서 씀
# 다른 스케일러들을 써보자! 
# MaxAbsScaler, PowerTransformer
# https://mkjjo.github.io/python/2019/01/10/scaler.html

import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

#데이터 전처리 (MinMax) 
print(np.max(x[0]))\

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

############ 과제 ##############################################################

# 다른 스케일러를 더 해보자
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer

# MaxAbsScaler -----------------------------------------------------------
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.
# scaler = MaxAbsScaler()

# PowerTransformer  -----------------------------------------------------------

# scaler = PowerTransformer(method='yeo-johnson')
scaler = PowerTransformer(method='box-cox')
# ValueError: The Box-Cox transformation can only be applied to strictly positive data
# box-cox 는 양수만 지원한다.
#################################################################################

scaler.fit(x)
x = scaler.transform(x) 

# 스탠다스스케일러 적용
print(np.max(x), np.min(x)) #9.933930601860268 -3.9071933049810337
print(np.max(x[0]))         #0.44105193260704206

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

# 스탠다스스케일러 적용
# loss, mae :  7.729104518890381 2.1026339530944824
# RMSE:  2.780127381279029
# R2:  0.9075276784607904

# 로버스트스케일러 적용
# loss, mae :  10.694147109985352 2.404524087905884
# RMSE:  3.2701909130372786
# R2:  0.8720534483580122

# 퀀타일트랜스포머 적용
# loss, mae :  9.90483570098877 2.18379282951355
# RMSE:  3.147194806108854
# R2:  0.8814969243270337