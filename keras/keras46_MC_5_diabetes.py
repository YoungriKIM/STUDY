#실습 19_1,2,3,4,5, earlystopping까지
#총 6개의 파일을 완성하시오

import numpy as np
from sklearn.datasets import load_diabetes #당뇨병 수준

#데이터 불러오고 , 이건 사이킷런의 데이터 불러오는 방법
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

#트레인 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle=True, random_state=66)

#데이터 전처리 (MinMaxScaler를 이용해서 , 기준은 x_train으로)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(42, input_shape=(10,)))
model.add(Dense(84))
model.add(Dense(84))
model.add(Dense(84))
model.add(Dense(42))
model.add(Dense(1))

#컴파일, 훈련 (Earlystopping 적용)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

modelpath = '../data/modelcheckpoint/k46_5_diabetes_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping, mc])

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss, mae: ', loss, mae)

y_predict = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)


#시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(18,6))

plt.subplot(2,1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['mae'], marker='.', c='red', label='mae')
plt.plot(hist.history['val_mae'], marker='.', c='blue', label='val_mae')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

#================================================
#6파일
# loss, mae:  3121.9990234375 46.306800842285156
# RMSE:  55.87484721026777
# R2:  0.5189554519135346