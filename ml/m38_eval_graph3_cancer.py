# 그래프로 그려서 확인해보자~

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
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss', 'error', 'auc'], eval_set=[(x_train, y_train), (x_test, y_test)])

aaa = model.score(x_test, y_test)
print('score: ',aaa)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('r2: ', acc)

print('----------------------------------------')
# 딥러닝 모델처럼(핏을 hist로 반환) evals를 반환해서 그것을 지표로 early stop을 할 수 없을까? > 그래프로도 그려보자
result = model.evals_result()
print(result)

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)


# 첫번째 그래프
fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('logloss')
plt.title('XGBoost logloss')
plt.show()

# 두번째 그래프
fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['error'], label = 'Train')
ax.plot(x_axis, result['validation_1']['error'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')
plt.show()

# 세번째 그래프
fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['auc'], label = 'Train')
ax.plot(x_axis, result['validation_1']['auc'], label = 'Test')
ax.legend()
plt.ylabel('auc')
plt.title('XGBoost auc')
plt.show()

# ========================================
# score:  0.9210526315789473
# r2:  0.9210526315789473

# ==================================================
# m36 metirc 3개
# [99]    validation_0-logloss:0.24113    validation_0-error:0.00879      validation_0-auc:0.99958
#         validation_1-logloss:0.30689    validation_1-error:0.07895      validation_1-auc:0.97891