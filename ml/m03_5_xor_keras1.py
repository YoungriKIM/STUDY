# XOR 문제로 만들어서 풀어보자 > 딥러닝으로 수정해서 해결해보자

from sklearn.svm import LinearSVC, SVC #LinearSVC가 개선된 SVC를 불러와서 쓰자. 3차원으로 접어서 선을 그었다고 생각해
import numpy as np
from sklearn.metrics import accuracy_score  #에큐러시를 메트릭스로 빼주는 녀석/ 회귀에서는 R2_score를 썼음
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]

#2. 모델
# model = LinearSVC()
# model = SVC()

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3.컴파일, 훈련
# model.fit(x_data, y_data)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=10)

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, '의 예측결과: ', y_pred)

# result = model.score(x_data, y_data)
# print('model.score: ', result)      # acc가 나올 것
result = model.evaluate(x_data, y_data)
print('model.score: ', result[1])     

# acc = accuracy_score(y_data, y_pred)  # 나오게 알아서 해결
# print('accuracy_score: ', acc)

# -----------------------------------------
# 히든 레이어 추가하기 전
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과:  [[0.50025   ]
#  [0.24241701]
#  [0.2886326 ]
#  [0.11481155]]
# 1/1 [==============================] - 0s 1ms/step - loss: 0.8688 - acc: 0.2500
# model.score:  0.25
