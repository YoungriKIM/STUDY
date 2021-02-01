# m10_2를 가져와서 여러 머신러닝 돌려서 비교 

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score     # KFold는 여기! / cross_val_score: 교차검증값(자른 것을 돌아가면서 검증)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 불러오기
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=311)

kfold = KFold(n_splits=5, shuffle=True)     # 5개로 나눠서 섞을거야

models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]


# for문 만들어보기
for i in np.arange(4):
    a = models[i]
    model = a 

    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(models[i], 'scores: ', scores)


#===========================================
# DNN 모델
# R2:  0.5189554519135346

# m10_4  KFold train/ val // test 일 때 ====================================
# LinearRegression() scores:  [0.3068361  0.41689003 0.49421515 0.60377324 0.58091078]
# KNeighborsRegressor() scores:  [0.46854488 0.2293146  0.22658192 0.40822179 0.36613351]
# DecisionTreeRegressor() scores:  [-0.30381381 -0.45253523  0.14078305 -0.11236748 -0.25352993]
# RandomForestRegressor() scores:  [0.35647406 0.35175897 0.5518484  0.37000861 0.40895365]