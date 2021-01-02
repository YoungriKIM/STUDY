from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
input1 = Input(shape=(5,)) 
aaa = Dense(5, activation='relu')(input1)
aaa = Dense(3)(aaa) 
aaa = Dense(4)(aaa)
outputs = Dense(2)(aaa) #히든 레이어의 이름을 아무렇게나 지어도 머신이 알아서 이름 정해서 한다.
model = Model(inputs = input1, outputs = outputs) 
model.summary() 

'''
# model = Sequential()
# #model.add(Dense(10, input_dim=1))
# model.add(Dense(5, activation='relu', input_shape=(5,)))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()

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
'''