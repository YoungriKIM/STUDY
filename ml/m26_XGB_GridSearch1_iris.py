# GridSearchCV + XGBRegressor 세트 5개 만들기/ pipeline 안 하고 위에서 자르고 들어가도 된다.

# 제공된 파라미터
# parameters = [
#     {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
#     {'colsample': [0.6, 0.9, 1], 'learning_rate' : [0.5, 0.01, 0.2], 'colsample_bytree': [0.5, 2, 0.1]},
#     {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'max_depth' : [2,4,8], 'colsample_bytree': [0.6, 0.9, 1]},
# ]

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#1. 데이터 지정, 전처리
dataset = load_iris()
x = dataset.data
y = dataset.target
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=519)

# 파라미터 지정
parameters = [
    {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
    {'colsample': [0.6, 0.9, 1], 'colsample_bytree': [0.5, 2, 0.1]},
    {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'colsample_bytree': [0.6, 0.9, 1]},
]

kfold = KFold(n_splits=5, shuffle=True)

#2. 모델(모델1)
model = GridSearchCV(XGBRegressor(n_jobs= 1), parameters, cv = kfold)

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, eval_metric='logloss', verbose=True, eval_set=[(x_train, y_train), (x_test, y_test)]) 
#eval_metric='logloss' 에러 잡아줌

#4. 평가(스코어)    
print('best_parameter: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('score: ', score)



#===========================================
# DNN 모델
# acc: 0.9736841917037964

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10)
# 최종정답률:  0.9473684210526315
# 74.222215초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_split=5)
# 최종정답률:  0.9385964912280702
# 11.309497초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.9385964912280702 입니다.
# Standard / 결과치는 0.9473684210526315 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.9298245614035088 입니다.

# =========================================================
# m22, 23
# score_1:  0.9333333333333333
# score_2:  0.9333333333333333
# 큰 차이 없음!

# =========================================================
# m24
# score_1:  0.9333333333333333
# score_2:  0.9333333333333333

# =========================================================
# n_jobs = 1            걸린 시간은 0.186884초 
# n_jobs = 2            걸린 시간은 0.206572초
# n_jobs = 4            걸린 시간은 0.208239초
# n_jobs = -1 = 8       걸린 시간은 0.198980초
# 확실하게 한 수가 압도적으로 좋다는 결론을 내릴 수는 없다.
# 이것도 역시 파라미터 튜닝이다

# =========================================================
# m26 GridSearch
# score:  0.9104654750410087
