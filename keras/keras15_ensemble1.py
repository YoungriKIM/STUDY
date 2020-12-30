# 다(2):다(2) 앙상블 모델

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
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle = True)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle = True)

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
dense2 = Dense(20, activation = 'relu')(dense2)
dense2 = Dense(20, activation = 'relu')(dense2)
dense2 = Dense(20, activation = 'relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate = 사슬처럼 엮다
from tensorflow.keras.layers import concatenate, Concatenate #소문자와 대문자를 나눠 놓은 이유가 있겠지..우선은 나중에
# Dense 와 Input도 레이어라서 계속 층을 쌓은 것처럼 concatenate도 할 수 있다.
# from keras.layers.merge import Concatenate  #옛날에는 이렇게 했는데 지금도 되긴 함
# from keras.layers import Concatenate # 마찬가지

merge1 = concatenate([dense1, dense2]) #소문자인 것 주의 #이 덴스1, 덴스2는 각 모델의 꼬다리임
middle1 = Dense(30)(merge1)
middle1 = Dense(15)(middle1)
middle1 = Dense(5)(middle1) #이 미들 레이어 안 쌓고 그냥 바로 보내도 된다

#모델 분기
output1 = Dense(7)(middle1)
output1 = Dense(14)(output1)
output1 = Dense(3)(output1) #얘네가 x1,y1꺼

output2 = Dense(6)(middle1)
output2 = Dense(12)(output2)
output2 = Dense(18)(output2)
output2 = Dense(3)(output2) #얘네가 x2,y2꺼

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2]) #2개 이상 들어갈 때는 []리스트로 묶어서 넣어줘라 이건 컴파일에도 마찬가지
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)
#x끼리 y끼리 같은 리스트에 묶어야 하는 것 주의

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)


print('model.metrics_names: ', model.metrics_names)
# model.metrics_names:  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']
# 모델에 사용한 메트릭스 보는 법 ! 써머리 보면서 비교해보렴
print('loss: ', loss)
#loss:  [11279.0693359375, 5046.35302734375, 6232.716796875, 5046.35302734375, 6232.716796875]
#loss 값이 5개인 이유: 첫번째는 1번 2번 모델의 합 즉 대표 loss, 두번째는 1번 모델의 로스(mse), 세번째는 2번 모델의 로스(mse), 네번째는 1번 모델의 메트릭스(mse), 다섯번째는 2번 모델의 메트릭스(mse)
#메트릭스를 mae로 바꾼다면 네번째, 다섯번째는 다르게 나온다.
#이렇게 loss:  [3612.236328125, 2139.316162109375, 1472.9200439453125, 37.99695587158203, 26.660287857055664]

y1_predict, y2_predict = model.predict([x1_test, x2_test])
# 이렇게 하면 (20,3)씩 2개가 나오겠지?

print('===========================')
print('y1_predict :\n', y1_predict) # 원 표시가 역슬래쉬임 \n은 줄바꿈임
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
# RMSE, R2에 넣기 위해서 무조건 넣는 것이 아니라 이렇게 빼서 해결 할 수 있다.