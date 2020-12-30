import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1 데이터 주고
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

#이러면 shape가 (3,100)이니 행렬를 바꿔주자

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test = train_test_split(x1, y1, x2, y2, train_size=0.8, shuffle=True)

#2 모델 구성
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(15, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(3, activation='relu')(input2)
dense2 = Dense(6, activation='relu')(dense2)
dense2 = Dense(9, activation='relu')(dense2)
dense2 = Dense(12, activation='relu')(dense2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(4)(merge1)
middel1 = Dense(8)(middle1)

output1 = Dense(15)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

output2 = Dense(9)(middle1)
output2 = Dense(6)(output1)
output2 = Dense(3)(output1)

model = Model(inputs = [input1, input2], outputs = [output1, output2])

model.summary()

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.8, verbose=0)

#평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
#모델에 쓰인 메트리스 이름 보기 : model.metrics_names
print('model.metrics_names: ', model.metrics_names)
# ['loss', 'dense_12_loss', 'dense_15_loss', 'dense_12_mae', 'dense_15_mae']
print('loss: ', loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])

print('==================================')
print('y1_predict: \n', y1_predict)
print('==================================')
print('y2_predict: \n', y2_predict)
print('==================================')

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE = (RMSE1+ RMSE2) /2
print('RMSE: ', RMSE)

from sklearn.metrics import r2_score
R2_1 = r2_score(y1_test, y1_predict)
R2_2 = r2_score(y2_test, y2_predict)
R2 = (R2_1+R2_2) /2
print('R2: ', R2)
