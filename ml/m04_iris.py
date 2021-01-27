# iris로 여러 머신러닝 모델을 돌리고 비교해보자

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#3. 컴파일, 훈련
model.fit(x_train,y_train)      # 머신러닝은 컴파일 없이 그냥 훈련이다.


#4. 평가, 예측
result = model.score(x_test,y_test)
print('model.score: ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score: ', acc)

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
