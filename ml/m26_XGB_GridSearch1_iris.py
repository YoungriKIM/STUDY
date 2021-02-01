# XGBoost로 GridSearchCV(RandomForest)로 세트 5개 만들기/ pipeline 안 하고 위에서 자르고 들어가도 된다.
# cross_val_score 써야겠찌?

# 제공된 파라미터
# parameters = [
#     {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
#     {'colsample': [0.6, 0.9, 1], 'learning_rate' : [0.5, 0.01, 0.2], 'colsample_bytree': [0.5, 2, 0.1]},
#     {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'max_depth' : [2,4,8], 'colsample_bytree': [0.6, 0.9, 1]},
# ]


# 하
# 는
# 중



import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#1. 데이터 지정, 전처리
dataset = load_iris()
x = dataset.data
y = dataset.target
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=519)

# 파라미터 지정
parameters = [
    {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
    {'colsample': [0.6, 0.9, 1], 'learning_rate' : [0.5, 0.01, 0.2], 'colsample_bytree': [0.5, 2, 0.1]},
    {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'max_depth' : [2,4,8], 'colsample_bytree': [0.6, 0.9, 1]},
]

#2. 모델(모델1)
model = GridSearchCV(RandomForestClassifier(), parameters)

#3. 컴파일ㄴ 훈련ㅇ

#4. 평가(스코어)    
score_1 = cross_val_score(model, x_train, y_train)
print('feature_names_1: \n', dataset.feature_names)
print('importances_1 : \n', model.feature_importances_)
print('score_1: ', score_1)

'''
# 중요도 그래프 그리기(솎기 전)
def plot_feature_importances_datasets(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_datasets(model)
# plt.show()


# 남길 개수 정하고 솎는 함수(수현)
def cut_columns(feature_importances, columns, number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()                 # 큰 수부터 앞으로 정렬
    temp = temp[:number]        # 내가 지정해준 개수만 반환
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])       # columns는 feature_names를 지정할 예정
    return result

# x데이터를 솎은 모델을 만들자(모델2)
x2 = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# 필요한 만큼만 위의 함수를 이용해 남기기
x2.drop(cut_columns(model.feature_importances_, dataset.feature_names, 2), axis=1, inplace=True)
# 내가 남긴 것들의 칼럼명 출력
print('feature_names_2: \n',cut_columns(model.feature_importances_, dataset.feature_names, 2))

# 모델2를 위한 전처리(x가 x2로 변경, random_state 동일하게 유지)
x2_train, x2_test, y_train, y_test = train_test_split(x2.values, y, test_size = 0.2, shuffle=True, random_state= 519)

#2. 모델1과 동일
model2 = XGBClassifier(n_jobs= 1) # cpu 코어를 모두 쓰겠다.

#3. 컴파일ㄴ 훈련ㅇ
model2.fit(x2_train, y_train)

#4. 평가(스코어)
score_2 = model2.score(x2_test, y_test)
print('importances_2 : \n', model2.feature_importances_)
print('score_2: ', score_2)


#===========================================
# DNN 모델
# acc: 0.9736841917037964

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10)
# 최종정답률:  0.9473684210526315
# 74.222215초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_split=5)
# 최종정답률:  0.9385964912280702
# 11.309497초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.9385964912280702 입니다.
# Standard / 결과치는 0.9473684210526315 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.9298245614035088 입니다.

# =========================================================
# m22, 23
# score_1:  0.9333333333333333
# score_2:  0.9333333333333333
# 큰 차이 없음!

# =========================================================
# m24
# score_1:  0.9333333333333333
# score_2:  0.9333333333333333

# =========================================================
# n_jobs = 1            걸린 시간은 0.186884초 
# n_jobs = 2            걸린 시간은 0.206572초
# n_jobs = 4            걸린 시간은 0.208239초
# n_jobs = -1 = 8       걸린 시간은 0.198980초
# 확실하게 한 수가 압도적으로 좋다는 결론을 내릴 수는 없다.
# 이것도 역시 파라미터 튜닝이다

# =========================================================
# m26 GridSearch
'''