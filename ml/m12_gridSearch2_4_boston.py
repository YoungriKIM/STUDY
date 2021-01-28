# m12 가져와서 모델: RandomForestClassifier 을 써서 gridsearchscore를 써보자

# 제공하는 파라미터
# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6,8,10,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1,2,4]}
# ]

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import timeit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 불러오기
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {'n_estimators' : [100,10], 'n_jobs' : [-1]},
    {'max_depth' : [6,8,10,12], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10], 'max_depth' : [6,8,10,12]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10]}
]

#2. 모델 구성
#--------------------------------------------------------------------------------
start_time = timeit.default_timer()  # 시작한 시각 체크
#--------------------------------------------------------------------------------

model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수: ', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률: ', r2_score(y_pred, y_test))

#--------------------------------------------------------------------------------
end_time = timeit.default_timer()  # 종료한 시각 체크
print('%f초 걸렸습니다.' % (end_time - start_time))
#--------------------------------------------------------------------------------

#===========================================
# DNN 모델
# R2:  0.8633197073667653

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_split=5)
# 최종정답률:  0.6734256301552818
# 30.139991초 걸렸습니다.