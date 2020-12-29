#실습
# x는 (100,5) 임의로 데이터 구성
# y는 (100,2) 임의로데이터 구성 하여 모델을 완성하시오
# 그 후에 predict의 일부값을 출력해보시오


import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(201,301), range(501,601)])
y = np.array([range(711, 811), range(1,101)])

# x_pred2 = np.array([[100],[401],[101],[301],[601]]) #내가 실습한 부분
x_pred2 = np.array([100,401,101,301,601])
# 이렇게 하면 트렌스포스가 안먹히는데 트렌스포스는 행과 열을 바꿔주는 코드이기 때문이다. 이것은 행렬이 아니다. 이럴때는 아래 코드를 이용하자
x_pred2 = x_pred2.reshape(1, 5) #이렇게 하면 [[100,401,101,301,601]]이 된다.

print(x.shape) #(5,100)
print(y.shape) #(2,100)

x = np.transpose(x)
y = np.transpose(y)
# x_pred2 = np.transpose(x_pred2) #내가 실습한 부분
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
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)
# validation을 넣어주지 않았을 때 나오지 않았으니 디폴트가 없다.(나중에 나오는 것도 있는데 우선은)

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
y_pred2 = model.predict(x_pred2) #가중치는 훈련하면서 이미 나왔기 때문에 predict는 몇번이나 해도 상관 없다.
print('y_pred2: ',y_pred2)
# y_predict2:  [[811.      100.99998]] 비슷하게 잘 나왔다~