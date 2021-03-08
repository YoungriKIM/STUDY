# 실습
# 1. 상단모델에 그리드 서치 또는 랜덤 서치로 튜닝한 모델 구성하고 최적의 R2 값과 피처임포턴스 구할 것
# 2. 위의 쓰레드 값으로 SelectFromModel 을 구해서 최적의 피처 갯수 구할 것
# 3. 위 피터 갯수로 데이터(피처)를 수정 (삭제)하여 그리드서치 또는 랜덤서치 적용하여 최적의 R2 값 구할 것

# 1번값과 2번값 비교

import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

'''
# 사전 모델 구성 (랜덤 서치로)--------------------------------------------------------
# 데이터 지정
x,y=load_boston(return_X_y=True)
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=23)
# 파라미터 지정
parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
]
# 모델 정의
model=RandomizedSearchCV(XGBRegressor(n_jobs=-1), parameters)
# 컴파일 ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose=1, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric='logloss')
# 평가(스코어)
print('best_parameter: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('score: ', score)
# score:  0.83568820629341

# xgoost 는 feature_importances로 피쳐별 중요도를 확인 할 수 있다.  ------------------------------
thresholds = np.sort(model.best_estimator_.feature_importances_)
print('thresholds : \n', thresholds)
# thresholds : 
#  [0.00309309 0.0043032  0.00861481 0.00916045 0.01516634 0.0202337
#  0.02152071 0.02285    0.02471725 0.03117503 0.04827791 0.22483993
#  0.56604755]


# SelectFromModel을 해보자 --------------------------------------------------------------------

best_model = model.best_estimator_

for thresh in thresholds :
    selection = SelectFromModel(best_model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape) 

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" % (thresh, select_x_train.shape[1], score*100))


# ============================================
# (404, 13)
# Thresh=0.007, n=13, R2: 85.36%
# (404, 12)
# Thresh=0.009, n=12, R2: 85.25%
# (404, 11)
# Thresh=0.010, n=11, R2: 84.93%
# (404, 10)
# Thresh=0.011, n=10, R2: 85.35%
# (404, 9)
# Thresh=0.011, n=9, R2: 83.77%
# (404, 8)
# Thresh=0.017, n=8, R2: 82.54%
# (404, 7)
# Thresh=0.023, n=7, R2: 82.38%
# (404, 6)
# Thresh=0.026, n=6, R2: 80.91%
# (404, 5)
# Thresh=0.032, n=5, R2: 78.74%
# (404, 4)
# Thresh=0.037, n=4, R2: 78.25%
# (404, 3)
# Thresh=0.050, n=3, R2: 71.92%
# (404, 2)
# Thresh=0.196, n=2, R2: 66.59%
# (404, 1)
# Thresh=0.571, n=1, R2: 43.27%

'''



# 사전 모델 구성 (그리드 서치로)--------------------------------------------------------
# 데이터 지정
x,y=load_boston(return_X_y=True)
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=23)
# 파라미터 지정
parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
]
# 모델 정의
model=GridSearchCV(XGBRegressor(n_jobs=-1), parameters)
# 컴파일 ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose=1, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric='logloss')
# 평가(스코어)
print('best_parameter: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('score: ', score)
# score:  0.83568820629341

# xgoost 는 feature_importances로 피쳐별 중요도를 확인 할 수 있다.  ------------------------------
thresholds = np.sort(model.best_estimator_.feature_importances_)
print('thresholds : \n', thresholds)
# thresholds : 
#  [0.00309309 0.0043032  0.00861481 0.00916045 0.01516634 0.0202337
#  0.02152071 0.02285    0.02471725 0.03117503 0.04827791 0.22483993
#  0.56604755]


# SelectFromModel을 해보자 --------------------------------------------------------------------

best_model = model.best_estimator_

for thresh in thresholds :
    selection = SelectFromModel(best_model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape) 

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" % (thresh, select_x_train.shape[1], score*100))


# ============================================
# (404, 13)
# Thresh=0.003, n=13, R2: 85.36%
# (404, 12)
# Thresh=0.008, n=12, R2: 85.25%
# (404, 11)
# Thresh=0.010, n=11, R2: 83.56%
# (404, 10)
# Thresh=0.012, n=10, R2: 84.72%
# (404, 9)
# Thresh=0.013, n=9, R2: 83.04%
# (404, 8)
# Thresh=0.015, n=8, R2: 82.73%
# (404, 7)
# Thresh=0.022, n=7, R2: 82.38%
# (404, 6)
# Thresh=0.029, n=6, R2: 80.91%
# (404, 5)
# Thresh=0.032, n=5, R2: 78.74%
# (404, 4)
# Thresh=0.034, n=4, R2: 74.61%
# (404, 3)
# Thresh=0.054, n=3, R2: 71.92%
# (404, 2)
# Thresh=0.267, n=2, R2: 66.59%
# (404, 1)
# Thresh=0.502, n=1, R2: 43.27%
