#데이터를 넘파이형태로 저장해보자

import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #(NN = 뉴럴 네트워크) ANN이면 인공지능 뉴럴 네트워크, RNN,DNN 등이 있다.

model = Sequential() #순차적으로 가는 모델을 구성할래
model.add(Dense(5, input_dim=1, activation='linear')) #add로 층을 추가할래 , 인풋은 시작하는 층임(1)
model.add(Dense(3, activation='linear'))#위층의 아웃풋이 이 녀석의 인풋이니까 인풋 쓸 필요가 없다
model.add(Dense(4))
model.add(Dense(1))
#시퀀셜이라서 맨 윗줄만 인풋을 써주고 다른 애들은 안써줘도 된다.

#3. 컴파일, 훈련 / fit은 훈련시키는 것임
model.compile(loss='mse', optimizer='adam')
#컴퓨터가 알아 듣게 설정해 주는 것이 주는 것이 compile이다.
#loss를 mse로 측정을 하고 그것을 위한 최적화는 adam을 써라 
model.fit(x,y, epochs=100, batch_size=1) #epochs는 선을 그으면서 학습하는 횟수임 100번 하자
#batch_size는 학습에 들어가는 데이터 사이즈임, 한 번에 몇개의 문제를 풀건지(사이즈가 클수록 빠르고 작을 수록 정확하다)
#이 시점에서 w와 b는 이미 생성이 된 것임

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
#우리는 이미 x랑 y로 훈련을 시켰는데 평가도 x,y로 하는게 제대로 된 평가가 나올리가 없다, 이제 다음에는 훈련할 데이터/평가할 데이터를 나눌 것이다
print("loss : ", loss) #""나 ''나 차이가 없다.

x_pred = model.predict([4]) #내가 예측하고 싶은 값을 넣는다. 이 경우에는 4를 넣었다. 이제 나오는 값은 x가 4일 때 y값이다.
print('result : ', x_pred)
