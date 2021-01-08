#18-earlystopping 파일 가져와서 사용

import numpy as np

from sklearn.datasets import load_boston 

#데이터 주고
dataset = load_boston()
x = dataset.data
y = dataset.target

#트레인이랑 테스트 분리 #트레인을 기준으로 minmax fit 할거라서
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.8, shuffle=True, random_state=66)

#데이터 전처리 (MinMax) #기준은 x_train
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
scaler.transform(x_val)

#모델 짜기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout #Dropout은 되도록이면 노드가 많은 곳에 붙인다.

inputs = Input(shape=(13,))
dense1 = Dense(56, activation='relu')(inputs)
dropout1 = Dropout(0.2)(dense1) #위 레이어의 노드에 20%를 off한다.
dense1 = Dense(28, activation='relu')(dropout1)
dense1 = Dense(56, activation='relu')(inputs)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(28, activation='relu')(dropout1)
dense1 = Dense(56, activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
outputs = Dense(1, activation='relu')(dropout1)

model = Model(inputs= inputs, outputs = outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

#early stopping 적용
from tensorflow.keras.callbacks import EarlyStopping #callback = 호출하다
early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto')

model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping])

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

# 18-earlystopping 
# loss, mae :  14.958937644958496 2.9046790599823
# RMSE:  3.86767836457503
# R2:  0.821028831848539

# 38-1 dropout 적용함(노드를 너무 줄였다는 뜻)
# loss, mae :  66.32543182373047 5.250315189361572
# RMSE:  8.144043242625342
# R2:  0.2064715304935698
