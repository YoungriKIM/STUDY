# OR 문제로 만들어서 풀어보자

from sklearn.svm import LinearSVC #supportVectormachine 선형으로 갈라지는 머신
import numpy as np
from sklearn.metrics import accuracy_score  #에큐러시를 메트릭스로 빼주는 녀석/ 회귀에서는 R2_score를 썼음

#1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,1]

#2. 모델
model = LinearSVC()

#3.훈련
model.fit(x_data, y_data)

#4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data, '의 예측결과: ', y_pred)

result = model.score(x_data, y_data)
print('model.score: ', result)      # acc가 나올 것

acc = accuracy_score(y_data, y_pred)
print('accuracy_score: ', acc)
#--------------------------------------------
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과:  [0 1 1 1]
# model.score:  1.0
# accuracy_score:  1.0