# nested: 중첩
# 해서 데이터를 잘라보자! m10-3 숙제의 답이 이거

# 제공하는 파라미터
# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6,8,10,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1,2,4]}                        #cpu를 몇개를 쓸것인지. -1: 전부 / 2: 2개
# ]

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target


kfold = KFold(n_splits=5, shuffle=True)

parameters = [      # 내가 원하는 파라미터를 딕셔너리 안에 키 밸류 형태로 넣어준다. 경우의 수는 곱해서 더한다.
    {'n_estimators' : [100,200], 'n_jobs' : [-1]},
    {'max_depth' : [6,8,10,12], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10], 'max_depth' : [6,8,10,12]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7,10]}
]

#2. 모델 구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold)

score = cross_val_score(model, x, y, cv=kfold)  # 이렇게 하면 5 * 5 하면 25번이 돌아간다.. 결과는 5개 나온다..?

print('교차검증점수: ', score)





'''
#3. 훈련
# model.fit(x_train, y_train)   # cross_val_score에 핏이 들어있어서 없어도 괜찮

#4. 평가, 예측
print('최적의 매개변수: ', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률: ', accuracy_score(y_pred, y_test))
'''
#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667
# 22.954975초 걸렸습니다.