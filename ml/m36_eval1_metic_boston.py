# XGBoost의 매트릭스에 두 개 이상 넣어보자!

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
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'logloss', 'mae'], eval_set=[(x_train, y_train), (x_test, y_test)])

aaa = model.score(x_test, y_test)
print('score: ',aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_red, y_test)  #### 이것보다 아래처럼 원데이터(y_test)가 앞으로 가게 넣어야 한다.
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

# 딥러닝 모델처럼(핏을 hist로 반환) evals를 반환해서 그것을 지표로 early stop을 할 수 없을까?
result = model.evals_result()
# print(result)


#==============================================
# score:  -5.430562540704325
# r2:  -5.430562540704325
# {'validation_0': OrderedDict([('rmse', [23.684231, 23.458986, 23.235727, 23.014912, 22.796055, 22.579586, 22.36478, 22.152565, 21.942448, 21.733944])]),
# 'validation_1': OrderedDict([('rmse', [23.485878, 23.261881, 23.04003, 22.820593, 22.603256, 22.388292, 22.174509, 21.964323, 21.755907, 21.548302])])}

#==============================================
# m36 metirc 3개
# [99]    validation_0-rmse:9.38555   validation_0-logloss:-792.32648  validation_0-mae:8.30738
#         validation_1-rmse:9.68606   validation_1-logloss:-797.14594  validation_1-mae:8.41665