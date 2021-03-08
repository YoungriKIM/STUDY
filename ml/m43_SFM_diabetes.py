# 이전 파일의 데이터를 당뇨병으로 만들어 봐라!
# R2 0.5 이상

import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score


# 사전 모델 구성 (랜덤 서치로)--------------------------------------------------------
# 데이터 지정
x,y=load_diabetes(return_X_y=True)
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
# score:  0.4252857569317312

# xgoost 는 feature_importances로 피쳐별 중요도를 확인 할 수 있다.  ------------------------------
thresholds = np.sort(model.best_estimator_.feature_importances_)
print('thresholds : \n', thresholds)
# thresholds : 
#  [0.040089   0.04096389 0.05135554 0.05267278 0.06337044 0.06596117
#  0.08236074 0.11042155 0.20787956 0.28492528]


# SelectFromModel을 해보자 --------------------------------------------------------------------

best_model = model.best_estimator_

for thresh in thresholds :
    selection = SelectFromModel(best_model, threshold=0.041, prefit=True)

    select_x_train = selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape) 

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" % (0.041, select_x_train.shape[1], score*100))


# ============================================

