# https://dacon.io/competitions/official/235626/codeshare/1555
# 을 해석함

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization

# 데이터 맨 처음 지정 ===================
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

# mnist 시각화해서 보기 ====================
# idx = 5
# img = train.loc[idx, '0':].values.reshape(28,28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s' % (idx, digit, letter))
# plt.imshow(img)
# plt.show()

# trian 데이터 지정 =======================
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28 , 1)

# print(x_train.shape)    #(2048, 28, 28, 1)
x_train = x_train/255

y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))
# print(len(y)) # 2048
# print(len(y.unique())) # 10
# unique(x) : 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환
# 즉, digit에 들어있는 수는 0~9까지 10가지임
for i , digit in enumerate(y):
    # print(i, digit)
    y_train[i, digit] = 1
# 위 for문으로 onehotencording을 해준 것임
# enumerate : 열거하다. range와 비슷하지만 순서와 담긴 내용을 함께 반환해줌
# 1039 4        > 1039행의 4번째 열에 1을
# 1040 5        > 1040행의 5번째 열에 1을
# 1041 8        > 1041행의 8번째 열에 1을

# 모델 구성 =========================
# print(x_train.shape[1:])    #(28, 28, 1)

def mymodel(x_train):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=5, input_shape=(x_train.shape[1:]), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 2, padding='same', activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(256, 2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 2, padding='same', activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    return model

# 컴파일, 훈련 =========================
model = mymodel(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, verbose=1, validation_split=0.2)

# 평가ㄴ 예측ㅇ =======================
# 예측에 사용할 test 데이터 지정
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255         # 흑백 이미지의 가장 큰 값은 255

sub = pd.read_csv('../data/csv/dacon3/submission.csv')
sub['digit'] = np.argmax(model.predict(x_test), axis=1)
print(sub.head())
# 잘 저장된 모습
#      id  digit
# 0  2049      6
# 1  2050      9
# 2  2051      6
# 3  2052      0
# 4  2053      3

# csv로 저장
sub.to_csv('../data/csv/dacon3/baseline.csv', index = False)
# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================
# baseline.csv > dacon score = 0.78921
