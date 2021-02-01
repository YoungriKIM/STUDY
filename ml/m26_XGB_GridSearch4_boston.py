# GridSearchCV + XGBRegressor 세트 5개 만들기/ pipeline 안 하고 위에서 자르고 들어가도 된다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 지정, 전처리
dataset = load_boston()
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
model = GridSearchCV(XGBRegressor(n_jobs=-1), parameters, cv = kfold)

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose=1, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric='logloss')

#4. 평가(스코어)
print('best_parameter: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('score: ', score)


# =========================================================
# m22, 23
# score_1:  0.873118986297473
# score_2:  0.872491444337033

# =========================================================
# m24
# score_1:  0.867857033635421
# score_2:  0.8471985667897188

# =========================================================
# m26 GridSearch
# score:  0.8587279778798291