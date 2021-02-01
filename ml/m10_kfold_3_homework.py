# 0128과제
# train test 나눈 다음 train만 kfold 하여 val 만들지 말고
# train, test로 5등분 kfold한 다음 잘린 train 안에서 train_test_split으로 val도 만들기

<<<<<<< HEAD
# m17에 답이 있다....
=======
>>>>>>> 00e481cd645c32239408a3bf2a28e6588d5c0aa8

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score     # KFold는 여기! / cross_val_score: 교차검증값(자른 것을 돌아가면서 검증)
from sklearn.metrics import accuracy_score

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 이름이 회귀스럽지만 분류모델이다!
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

kfold = KFold(n_splits=5, shuffle=True, random_state=311)     # 5개로 나눠서 섞을거야

#2. 모델 구성
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

for train_index, test_index in kfold.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)

    train_index, val_index = train_test_split(train_index, train_size=0.8, shuffle=True, random_state=311)
    x_train, x_val, x_test = x[train_index], x[val_index], x[test_index]

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

for train_index, test_index in kfold.split(y):
    # print("TRAIN:", train_index, "TEST:", test_index)

    train_index, val_index = train_test_split(train_index, train_size=0.8, shuffle=True, random_state=311)
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

scores = cross_val_score(model, x, y, cv=kfold)
print('scores: ', scores)

#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

# MinMaxScaler 일 때 ====================================
# model = LinearSVC()
# result:  0.9666666666666667
# accuracy_score:  0.75

# model = SVC()
# result:  1.0
# accuracy_score:  1.0

# model = KNeighborsClassifier()
# result:  1.0
# accuracy_score:  1.0

# model = LogisticRegression()
# result:  1.0
# accuracy_score:  1.0

# model = DecisionTreeClassifier()
# result:  0.9333333333333333
# accuracy_score:  1.0

# model = RandomForestClassifier()
# result:  0.9333333333333333
# accuracy_score:  0.75

