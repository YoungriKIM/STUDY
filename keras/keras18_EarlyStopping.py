# Earlystopping

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
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(13,))
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(inputs)
dense1 = Dense(56, activation='relu')(inputs)
dense1 = Dense(28, activation='relu')(dense1)
dense1 = Dense(56, activation='relu')(dense1)
outputs = Dense(1, activation='relu')(dense1)

model = Model(inputs= inputs, outputs = outputs)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

#early stopping 적용
from tensorflow.keras.callbacks import EarlyStopping #callback = 호출하다
early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto')
#patience = 로스값이 최소 값보다 더 떨어지지 않는 것을 n번 참겠다. (연속으로) #로스는 작아야 하니까 min 인데 outo도 상관 없다
# 지금 멈추는 epo 지점은 최소값을 지나지 않은 20번째 자리이다.

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

#전처리 전
# loss, mae :  19.918365478515625 3.237553119659424
# RMSE:  4.462999550425985
# R2:  0.7616934074949592

#전처리 후 x = x/711. > 칼럼의 특성에 맞는 전처리를 한 것이 아님
# loss, mae :  11.487313270568848 2.5379178524017334
# RMSE:  3.3892934685148166
# R2:  0.8625639325151193

# MinMaxscaler 적용 후 > x 통채로 전처리 한 것
# loss, mae :  10.720622062683105 2.2357163429260254
# RMSE:  3.2742366375542535
# R2:  0.871736673764582

# 제대로 전처리 (validation_split)
# loss, mae :  12.26821517944336 2.707951307296753
# RMSE:  3.5026018052177164
# R2:  0.8532210066127037

# 제대로 전처리 (validation_data) 
# loss, mae :  14.958937644958496 2.9046790599823
# RMSE:  3.86767836457503
# R2:  0.821028831848539

