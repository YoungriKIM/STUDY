import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(201,301), range(501,601)])
y = np.array([range(711, 811), range(1,101)])

x_pred2 = np.array([100,401,101,301,601])
x_pred2 = x_pred2.reshape(1, 5) #이렇게 하면 [[100,401,101,301,601]]이 된다.

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
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)
#verbose는 훈련과정을 보여줄지 정하는 것인데. 1으로 정하면 다 보이는 건데 만약 머신이 나한테 이 과정을 보여주려면 화면에 띄우느라 딜레이가 더 걸린다. 선택은 알아서\
# evaluate에 써도 된다.
'''
verbose=장황한
+ verobose = 0 : 훈련과정이 아예 안 보임
+ verobose = 1 : 훈련과정이 모두 보임 (디폴트)
64/64 [==============================] - 0s 3ms/step - loss: 7155.4707 - mae: 58.6185 - val_loss: 818.9250 - val_mae: 21.0954
Epoch 2/100
+ verobose = 2 :
64/64 - 0s - loss: 221494.2656 - mae: 401.2755 - val_loss: 107248.1562 - val_mae: 283.1143
Epoch 2/100
+ verobose = 3 :
Epoch 2/100
+ verobose = 4 : 3과 동일
'''


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