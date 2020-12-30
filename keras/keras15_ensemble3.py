# 다(2):다(3) 로 바꾸자. 분기를 안한다는 힌트


import numpy as np

#1. 데이터 제공
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411,511), range(100,200)])

y1 = np.array([range(711, 811), range(1,101), range(201,301)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(601,701), range(811,911), range(1100,1200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.8, shuffle = False)
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, train_size=0.8, shuffle = False)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_test.shape)

print(y1_train.shape)
print(y1_test.shape)
print(y2_train.shape)
print(y2_test.shape)
print(y3_train.shape)
print(y3_test.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)
# output1 = Dense(3)(dense1)
# 엮은 다음에 아웃풋이 나와야 하기 때문에 지금 있으면 안됨

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate = 사슬처럼 엮다
from tensorflow.keras.layers import concatenate, Concatenate #소문자와 대문자를 나눠 놓은 이유가 있겠지..우선은 나중에
# Dense 와 Input도 레이어라서 계속 층을 쌓은 것처럼 concatenate도 할 수 있다.
# from keras.layers.merge import Concatenate  #옛날에는 이렇게 했는데 지금도 되긴 함
# from keras.layers import Concatenate # 마찬가지

merge1 = concatenate([dense1, dense2]) #소문자인 것 주의 #이 덴스1, 덴스2는 각 모델의 꼬다리임
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1) #이 미들 레이어 안 쌓고 그냥 바로 보내도 된다

#모델 분기
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1) #얘네가 y1꺼

output2 = Dense(30)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2) #얘네가 y2꺼

output3 = Dense(30)(middle1)
output3 = Dense(5)(output3)
output3 = Dense(15)(output3)
output3 = Dense(3)(output3) #얘네가 y3꺼

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2,output3])
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=10, batch_size=1, validation_split=0.2, verbose=0)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)


print('model.metrics_names: ', model.metrics_names)
#써머리 보면서 비교해보렴
print('loss: ', loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])

print('===========================')
print('y1_predict :\n', y1_predict) # 원 표시가 역슬래쉬임 \n은 줄바꿈임
print('===========================')
print('y2_predict :\n', y2_predict)
print('===========================')
print('y3_predict :\n', y3_predict)


#y1_test, y2_test와 비교하자 RMSE랑 R2로
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
RMSE = (RMSE1 + RMSE2 + RMSE3) /3
print('RMSE1: ',RMSE1)
print('RMSE2: ',RMSE2)
print('RMSE3: ',RMSE3)
print('RMSE: ', RMSE)


from sklearn.metrics import r2_score

R2_1= r2_score(y1_test, y1_predict)
R2_2= r2_score(y2_test, y2_predict)
R2_3= r2_score(y3_test, y3_predict)
R2 = (R2_1+R2_2+R2_3) /3
print('R2_1: ', R2_1)
print('R2_2: ', R2_2)
print('R2_3: ', R2_3)
print('R2: ', R2)

#아래 부분 해결했는데 다시 복습
x_check = np.array([[100, 401, 101]]), np.array([[201,511,200]])
y_check = model.predict(x_check)
print('y_check: ',y_check)
