#1:다 mlp
#실습 인풋이1 아웃풋이3으로 해서 수정하고 튠까지｜ 이런 모델을 권장하진 않지만 돌아가긴 한다.

import numpy as np

x = np.array([range(100)])
y = np.array([range(711, 811), range(1,101), range(201,301)])
x_pred2 = np.array([[100]])

print(x.shape) #(1,100)
print(y.shape) #(3,100)

x = np.transpose(x)
y = np.transpose(y)
# print(x)
print(x.shape) #(100,1)
print(y.shape) #(100,3)
print(x_pred2.shape) #(1,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)  #(80,1)
print(y_train.shape)  #(80,3)

#모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(50,  activation='relu'))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(3)) 

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=150, batch_size=1, validation_split=0.2, verbose=3)

loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

y_pred2 = model.predict(x_pred2)
print('y_pred2: ', y_pred2)