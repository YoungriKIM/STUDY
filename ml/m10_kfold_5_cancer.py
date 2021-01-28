# m10_2를 가져와서 여러 머신러닝 돌려서 비교 

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score     # KFold는 여기! / cross_val_score: 교차검증값(자른 것을 돌아가면서 검증)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 이름이 회귀스럽지만 분류모델이다! 2진분류만 하는 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터 불러오기
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=311)

kfold = KFold(n_splits=5, shuffle=True)     # 5개로 나눠서 섞을거야

models = [LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]


# for문 만들어보기
for i in np.arange(6):
    a = models[i]
    model = a 

    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(models[i], 'scores: ', scores)


#===========================================
# DNN 모델
# acc: 0.9736841917037964

# m10_4  KFold train/ val // test 일 때 ====================================
# LinearSVC() scores:  [0.92307692 0.87912088 0.84615385 0.92307692 0.93406593]
# SVC() scores:  [0.93406593 0.87912088 0.91208791 0.84615385 0.9010989 ]
# KNeighborsClassifier() scores:  [0.91208791 0.94505495 0.93406593 0.95604396 0.89010989]
# LogisticRegression() scores:  [0.93406593 1.         0.91208791 0.89010989 0.94505495]
# DecisionTreeClassifier() scores:  [0.87912088 0.94505495 0.96703297 0.96703297 0.95604396]
# RandomForestClassifier() scores:  [0.97802198 0.94505495 0.94505495 0.96703297 0.97802198]