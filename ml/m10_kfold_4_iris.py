# m10_2를 가져와서 여러 머신러닝 돌려서 비교 

import numpy as np
from sklearn.datasets import load_iris 
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
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=311)

kfold = KFold(n_splits=5, shuffle=True)     # 5개로 나눠서 섞을거야

models = [LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]


# for문 만들어보기 (아래 부분)
for i in np.arange(6):
    a = models[i]
    model = a 

    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(models[i], 'scores: ', scores)


#===================================================
#2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print('scores: ', scores)
#===================================================


#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

# m10_4  KFold train/ val // test 일 때 ====================================
# LinearSVC() scores:  [0.91666667 0.91666667 1.         1.         0.91666667]
# SVC() scores:  [0.875      0.91666667 0.91666667 0.95833333 0.95833333]
# KNeighborsClassifier() scores:  [0.95833333 1.         0.95833333 0.95833333 1.        ]
# LogisticRegression() scores:  [0.91666667 1.         0.95833333 1.         0.91666667]
# DecisionTreeClassifier() scores:  [0.95833333 1.         0.875      1.         0.95833333]
# RandomForestClassifier() scores:  [0.95833333 0.95833333 1.         0.95833333 0.875     ]
