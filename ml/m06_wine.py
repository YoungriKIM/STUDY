# wine 로 여러 머신러닝 모델을 돌리고 비교해보자
# 모델.스코어와 에큐러시.스코어가 같은지 확인!
# m04~m06은 분류모델. m07,m08은 회귀다~ 분류는 뒤에 Classifier, 회귀는 Regressor

#깨알 메모: sklearn이 tensorflow보다 먼저 만들어졌다.

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # 이름이 회귀스럽지만 분류모델이다!
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score: ', acc)

#======================================================
# 딥러닝 모델
# acc:  0.9722222089767456

#======================================================
# 머신러닝으로 돌리기
# model = LinearSVC()
# model.score:  0.9722222222222222
# accuracy_score:  0.9722222222222222

# model = SVC()
# model.score:  0.9722222222222222
# accuracy_score:  0.9722222222222222

# model = KNeighborsClassifier()
# model.score:  0.9444444444444444
# accuracy_score:  0.9444444444444444

# model = LogisticRegression()
# model.score:  0.9722222222222222
# accuracy_score:  0.9722222222222222

# model = DecisionTreeClassifier()
# model.score:  1.0
# accuracy_score:  1.0

# model = RandomForestClassifier()
# model.score:  0.9722222222222222
# accuracy_score:  0.9722222222222222