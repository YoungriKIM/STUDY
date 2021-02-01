# pipeline 으로 스케일링 + 알고리즘 하고 GridSearchCV으로 모델을 지정해주자!

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline    # pipeline 과 make_pipeline 은 쓰는 방법이 다를 뿐 비슷한 녀석들이다.

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

# 1 make_pipeline 용법 =============================================================================
# parameters = [
#     {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['linear']},                                      # 4번
#     {'svc__C' : [1,10,100], 'svc__kernel':['rdf'], 'svc__gamma':[0.001, 0.0001]},                  # 6번
#     {'svc__C' : [1,10,100,1000], 'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}          # 8번   > 18번을 5번 > 90번
# ]

# #2. 모델
# pipe = make_pipeline(MinMaxScaler(), SVC())  
# #                   ---------------
# 2 Pipe 용법 =============================================================================
parameters = [
    {'mal__C' : [1,10,100,1000], 'mal__kernel' : ['linear']},                                      # 4번
    {'mal__C' : [1,10,100], 'mal__kernel':['rdf'], 'mal__gamma':[0.001, 0.0001]},                  # 6번
    {'mal__C' : [1,10,100,1000], 'mal__kernel':['sigmoid'], 'mal__gamma':[0.001, 0.0001]}          # 8번   > 18번을 5번 > 90번
]

#2. 모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('mal', SVC())]) 
#                          ---------------
#=======================================================================================
#  왜 이렇게 파이프라인으로 스케일링이랑 모델을 합춰 줘?
# : kfold로 나눠서 돌릴 때 마다 train 과 test의 위치와 값이 달라지기 때문에 kfold 값이 5일 때 그 5번에 알맞게 맞춘 스케일링을 하기 위해서야
# + svc__ 파이프를 쓸 때는 이렇게 앞에 이름과 _을 두번 쓰는 것이 약속된 용어야
#=======================================================================================

#2.모델 구성
model = GridSearchCV(pipe, parameters, cv=5)
# model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('결과치는', results, '입니다.')


#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667
# 22.954975초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667
# 7.167455초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.9666666666666667 입니다.
# Standard / 결과치는 0.9666666666666667 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.9666666666666667 입니다.