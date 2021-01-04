#acc: 1.0 나올 때 까지 튜닝

import numpy as np
from sklearn.datasets import load_iris #회귀의 대표 모델은 보스턴이랑 당뇨병, 이중분류는 유방암, 다중분류는 아이리스임

# x, y = load_iris(return_x_y=True) #이렇게 x랑 y로 데이터를 줘도 된다. *다시 해보기

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape) #(150,4)
    # :Attribute Information:
    #     - sepal length in cm
    #     - sepal width in cm
    #     - petal length in cm
    #     - petal width in cm > 칼럼이 4개
    #     - class:
    #             - Iris-Setosa
    #             - Iris-Versicolour
    #             - Iris-Virginica > 결과y가 3개. 다중분류이다.

# print(x[:5])
# print(y) #0,1,2가 50개씩 순서대로 있기때문에 아래 데이터 분리에서 shuffle를 해주지 않으면 성능이 떨어질 수 밖에 없다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#원핫인코딩. y에 대한 전처리를 해주자. 이 데이터는 3개의 결과로 나와야 한다. 0: 1,0,0 / 1: 0,1,0 / 2: 0,0,1
from tensorflow.keras.utils import to_categorical #이게 텐서플로우 2.0의 방법이다.
# from keras.utils.np_utils import to_categorical #이게텐서플로우 1.0방식. 이것도 가능하긴 하다.

y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test) #트레인테스트스플릿 하기 전에 원핫인코딩을 해도 된다. 해보자
# print(y_train) #[1. 0. 0.] : 0 / [0. 1. 0.] : 1 / [0. 0. 1.] : 2
# print(y_train.shape) #(96,3)이 되었다.

print(x.shape) #(150,4)
print(y.shape) #(150,3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(120, activation='relu', input_shape=(4,)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(3, activation='softmax')) #다중분류는 softmax, 이진분류는 sigmoid / softmax로 나온 값은 모두 합치면 1이 된다. 이 중 가장 큰 값을 받은 애가 선택이 된다.
#아웃풋레이어의 노드를 3개로 잡아야 한다. 분류의 개수가 3개니까
#그러니 y를 원핫인코딩을 하여 3개로 나눠주여야 한다.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=2, callbacks=earlystopping)

loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)

# y_predict = model.predict(x_test[-5:-1])
# print('y_predict: ', y_predict)
# print('y_test[-5:-1]: ', y_test[-5:-1])

print('===========================')
# loss:  [0.12436151504516602, 0.9666666388511658, 0.04672175273299217]
