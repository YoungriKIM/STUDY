# m37 가져와서 씀
# XGBoost의 모델을 저장하자!
# model.save_model


import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터 
# x, y = load_boston(return_X_y=True) # 이것도 가능함 알아보고 찍어보기
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

#2. 모델
model = XGBRegressor(n_estimators=100, learning_rate=0.01, n_jobs=-1)
# n_estimator = 트리의 개수

#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=20)
# early_stopping_rounds= n  몇 번 돌고 끝낼래?

#4. 평가
aaa = model.score(x_test, y_test)
print('score: ',aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_red, y_test)  #### 이것보다 아래처럼 원데이터(y_test)가 앞으로 가게 넣어야 한다.
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

# 딥러닝 모델처럼(핏을 hist로 반환) evals를 반환해서 그것을 지표로 early stop을 할 수 없을까?
result = model.evals_result()
# print(result)

#--------------------------------------------------------------------------------------------------------
# 파이썬에서 제공하는 저장기능을 쓰자!

# 1 -----------------------------
import pickle
# # 저장
# pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb'))     # write binary
# print('=====save complete=====')

# # 불러오기
# model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb'))     # read binary
# print('======read complete=====')
# r22 = model2.score(x_test, y_test)
# print('score2: ', r22)

# 2 -----------------------------
import joblib
# # 저장
# joblib.dump(model, '../data/xgb_save/m40.joblib.dat')

# # 불러오기
# model3 = joblib.load('../data/xgb_save/m40.joblib.dat')
# print('======read complete=====')
# r23 = model3.score(x_test, y_test)
# print('score4: ', r23)

# 3 -----------------------------
#xgb 자체에 있는 모델 세이브를 쓰자

#저장
# model.save_model('../data/xgb_save/m41.xgb.model')

#불러오기
model4 = XGBRegressor()
model4.load_model('../data/xgb_save/m41.xgb.model')
print('======read complete=====')
r24 = model4.score(x_test, y_test)
print('score4: ', r24)




#==============================================
# score:  -5.430562540704325
# r2:  -5.430562540704325
# {'validation_0': OrderedDict([('rmse', [23.684231, 23.458986, 23.235727, 23.014912, 22.796055, 22.579586, 22.36478, 22.152565, 21.942448, 21.733944])]),
# 'validation_1': OrderedDict([('rmse', [23.485878, 23.261881, 23.04003, 22.820593, 22.603256, 22.388292, 22.174509, 21.964323, 21.755907, 21.548302])])}

#==============================================
# m36 metirc 3개
# [99]    validation_0-rmse:9.38555   validation_0-logloss:-792.32648  validation_0-mae:8.30738
#         validation_1-rmse:9.68606   validation_1-logloss:-797.14594  validation_1-mae:8.41665