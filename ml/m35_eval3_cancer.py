# 얼리스탑핑 하기 전에 쓸 지표를 먼저 알아야지 / 이중분류

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터 
dataset = load_breast_cancer()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

#2. 모델
model = XGBClassifier(n_estimators=100, learning_rate=0.01, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='logloss', eval_set=[(x_train, y_train), (x_test, y_test)])

aaa = model.score(x_test, y_test)
print('score: ',aaa)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('r2: ', acc)

# 딥러닝 모델처럼(핏을 hist로 반환) evals를 반환해서 그것을 지표로 early stop을 할 수 없을까?
result = model.evals_result()
print(result)   # hist 처럼 결과가 쭉~ 나온다!

# ========================================
# score:  0.9210526315789473
# r2:  0.9210526315789473