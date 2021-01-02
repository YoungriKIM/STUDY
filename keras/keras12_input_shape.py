import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(201,301), range(501,601)])
y = np.array([range(711, 811), range(1,101)])

x_pred2 = np.array([100,401,101,301,601])
x_pred2 = x_pred2.reshape(1, 5)

print(x.shape) #(5,100)
print(y.shape) #(2,100)

x = np.transpose(x)
y = np.transpose(y)
# print(x)
print(x.shape) #(100,5)
print(y.shape) #(100,2)
print(x_pred2.shape) #(1,5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)  #(80,5)
print(y_train.shape)  #(80,2)

#모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim=5))
model.add(Dense(10, input_shape=(5,))) #정상적으로 돌아가는데 즉 input_dim과 같은 것이다. 여기서 5은 컬럼(열)의 갯수다, 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=0)

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

###
y_pred2 = model.predict(x_pred2) 
print('y_pred2: ',y_pred2)