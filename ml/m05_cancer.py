# cancer 로 여러 머신러닝 모델을 돌리고 비교해보자
# 모델스코어와 에큐러시스코어가 같은지 확인!

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
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
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#트레인 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)

#데이터 전처리 (MinMaxScaler를 이용해서 , 기준은 x_train으로)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#컴파일, 훈련
model.fit(x_train, y_train)

#평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print('accuary_score: ', acc)


#===========================================
# DNN 모델
# acc: 0.9736841917037964

# MinMaxScaler 일 때 ====================================
# model = LinearSVC()
# model.score:  0.9736842105263158
# accuary_score:  0.9736842105263158

# model = SVC()
# model.score:  0.9736842105263158
# accuary_score:  0.9736842105263158

# model = KNeighborsClassifier()
# model.score:  0.956140350877193
# accuary_score:  0.956140350877193

# model = LogisticRegression()
# model.score:  0.9649122807017544
# accuary_score:  0.9649122807017544

# model = DecisionTreeClassifier()
# model.score:  0.9298245614035088
# accuary_score:  0.9298245614035088

# model = RandomForestClassifier()
# model.score:  0.9649122807017544
# accuary_score:  0.9649122807017544
