# diabetes 로 여러 머신러닝 모델을 돌리고 비교해보자
# 모델스코어와 에큐러시스코어가 같은지 확인!

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score    # 분류는 애큐러시 스코어, 회귀는 알투스코어

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

#트레인 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)

#데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
# model = LinearRegression()
# model = KNeighborsRegressor()
model = DecisionTreeRegressor()
# model = RandomForestRegressor()

#컴파일, 훈련
model.fit(x_train, y_train)

#평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2_score: ', r2)


#===========================================
# DNN 모델
# R2:  0.5189554519135346

# ML 일 때 ====================================
# model = LinearRegression()
# model.score:  0.5063891053505036
# r2_score:  0.5063891053505036

# model = KNeighborsRegressor()
# model.score:  0.3741821819765594
# r2_score:  0.3741821819765594

# model = DecisionTreeRegressor()
# model.score:  -0.1902922738481716
# r2_score:  -0.1902922738481716

# model = RandomForestRegressor()
# model.score:  0.375549881085699
# r2_score:  0.375549881085699