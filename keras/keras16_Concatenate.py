# 다(2):다(2) 앙상블 모델 concatenate를 Concatenate로

import numpy as np

#1. 데이터 제공
x1 = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(711, 811), range(1,101), range(201,301)])

x2 = np.array([range(101, 201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle = False)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle = False)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate(axis=1)([dense1, dense2])
#concatenate가 아니라 이번에는 Conctenate로 썼는데 (axis=)도 추가해줬다. axis는 1일 때 가장 높은 차원, 0일 때 가장 높은 것에서 두번째, -1일 때 가장 낮은 차원
#사용법만 조금 다를 뿐 성능은 똑같다.
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

#모델 분기
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1) #얘네가 y1꺼

output2 = Dense(30)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2) #얘네가 x2,y2꺼

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2]) #2개 이상 들어갈 때는 []리스트로 묶어서 넣어줘라 이건 컴파일에도 마찬가지
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=0)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)


print('model.metrics_names: ', model.metrics_names)
print('loss: ', loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])

print('===========================')
print('y1_predict :\n', y1_predict)
print('===========================')
print('y2_predict :\n', y2_predict)
print('===========================')

#y1_test, y2_test와 비교하자 RMSE랑 R2로
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE = (RMSE1 + RMSE2) /2
print('RMSE1: ',RMSE1)
print('RMSE2: ',RMSE2)
print('RMSE: ', RMSE)


from sklearn.metrics import r2_score

R2_1= r2_score(y1_test, y1_predict)
R2_2= r2_score(y2_test, y2_predict)
R2 = (R2_1+R2_1) /2
print('R2_1: ', R2_1)
print('R2_2: ', R2_2)
print('R2: ', R2)

#이 앙상블 모델이 무조건 좋다는 것은 아니다. 이런 방법이 있다는 것이고 모든 모델이 좋은지는 직접 하고 평가하자