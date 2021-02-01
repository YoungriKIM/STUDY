# 실습! load_boston
# 스케일러는 MinMax, Standard 네 맘대로
# RandomSearchCV, GridSearchCV + Pipeline, make_pipeline 을 엮어라
# 모델은 RandomForest

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
dataset = load_boston()
x = dataset.data
y = dataset.target

# 전처리: 트레인 테스트 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

# 파라미터 지정 _ 내가 지정한 알고리즘에 맞는 파라미터를 써야한다는 점 주의!!
# 1 make_pipeline 용법 =============================================================================
parameters = [
    {'randomforestregressor__n_estimators' : [50,100,200]},
    {'randomforestregressor__max_depth' : [6,8,10,12]},
    {'randomforestregressor__min_samples_leaf' : [3,5,7,10]},
    {'randomforestregressor__min_samples_leaf' : [3,5,7,10]},
    {'randomforestregressor__n_jobs' : [-1,2,4,8]}
]

# 파이프로 엮기
pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())

# 2 Pipe 용법 =============================================================================
# parameters = [
#     {'a__n_estimators' : [50,100,200]},
#     {'a__max_depth' : [6,8,10,12]},
#     {'a__min_samples_leaf' : [3,5,7,10]},
#     {'a__min_samples_leaf' : [3,5,7,10]},
#     {'a__n_jobs' : [-1,2,4,8]}
# ]

# # 2. 모델
# pipe = Pipeline([('scaler', MinMaxScaler()), ('a', RandomForestRegressor())]) 
#=======================================================================================

#2.모델 구성
# model = GridSearchCV(pipe, parameters, cv=5)
model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과치는', results, '입니다.')


#===========================================
# DNN 모델
# R2:  0.8633197073667653

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_split=5)
# 최종정답률:  0.6734256301552818
# 30.139991초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestRegressor(n_jobs=-1)
# 최종정답률:  0.7078922745367211
# 8.782590초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.7287531921601408 입니다.
# Standard / 결과치는 0.7248061849054493 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.7396423200556946 입니다.