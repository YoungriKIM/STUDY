# XGBoost로 Randomsearch(RandomForest)로 세트 5개 만들기/ pipeline 안 하고 위에서 자르고 들어가도 된다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance, XGBRegressor  # xgboost에 이미 들어있따.
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 지정, 전처리
dataset = load_wine()
x = dataset.data
y = dataset.target
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=519)

# 파라미터 지정
parameters = [
    {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
    {'colsample': [0.6, 0.9, 1], 'colsample_bytree': [0.5, 2, 0.1]},
    {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'colsample_bytree': [0.6, 0.9, 1]},
]

kfold = KFold(n_splits=5, shuffle=True, random_state=311)

#2. 모델(모델1)
model = RandomizedSearchCV(XGBClassifier(n_jobs=-1), parameters, cv = kfold)

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='mlogloss', verbose = True)
# 워닝에서 logloss말고 mlogloss쓰라고 해서 바꿔 씀 / mlogloss : Multiclass logloss

#4. 평가(스코어)    
print('best_parameter: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('score: ', score)


# =========================================================
# m21,23
# score_1:  0.9722222222222222
# score_2:  0.9722222222222222
# m24 위의 스코어와 동일

# =========================================================
# m26 GridSearch
# score:  0.9444444444444444

# m27 RandomizedSearchCV
# score:  0.9444444444444444
