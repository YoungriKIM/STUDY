# 머신러닝으로 디폴트 잡으려고 츄라이 함
# ML_try_0202_1 업그레이드

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 지정
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

x = train.drop(['id','digit','letter'], axis=1).values
y = train['digit'].values
all_test = test.drop(['id', 'letter'], axis=1).values


# y = train['digit']
# y2 = np.zeros((len(y), len(y.unique())))
# for i , digit in enumerate(y):
#     y2[i, digit] = 1

# print(x.shape)      #(2048, 784)
# print(y.shape)      #(2048, )

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=519)

# 파라미터 지정
# parameters = [
#     {'n_estimators' : [200,500,1000], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
#     {'colsample': [0.6, 0.9, 1], 'learning_rate' : [0.01, 0.001, 0.002], 'colsample_bytree': [0.5, 2, 0.1]},
#     {'n_estimators' : [500, 1000, 1200], 'learning_rate' : [0.01, 0.001, 0.005], 'max_depth' : [3,4,6], 'colsample_bytree': [0.6, 0.9, 1]},
# ]
# parameters = [{'n_estimators' : [10], 'learning_rate' : [0.01], 'max_depth' : [6], 'colsample_bytree': [1]}]

kfold = KFold(n_splits=5, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

#--------------------------------------------------------------------------------------------------------
# #2. 모델
model = XGBClassifier(n_estimators = 100, learning_rate = 0.01, max_depth = 6, colsample_bytree = 1)

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose=True, eval_metric = ['mlogloss'], eval_set=[(x_train, y_train), (x_test, y_test)])

#4. 평가
score = model.score(x_test, y_test)
print('score: ', score)

# --------------------------------------------------------------------------------------------------------
# 중요도 그래프 그리기(솎기 전)
def plot_feature_importances_datasets(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), np.arange(0, 784))
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_datasets(model)
# plt.show()
# --------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------
x2 = train.drop(['id','digit','letter'], axis=1)
x2.drop(cut_columns(model.feature_importances_, np.arange(0, 784), 4), axis=1, inplace=True).values

x2_train, x2_test, y_train, y_test = train_test_split(x2.values, y, test_size = 0.2, shuffle=True, random_state= 519)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
all_test = scaler.transform(all_test)

# #2. 모델
model = XGBClassifier(n_estimators = 100, learning_rate = 0.01, max_depth = 6, colsample_bytree = 1)

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose=True, eval_metric = ['mlogloss'], eval_set=[(x_train, y_train), (x_test, y_test)])

#4. 평가
score = model.score(x_test, y_test)
print('score2: ', score)


# 모델 pickle로 저장
# pickle.dump(model, open('../data/xgb_save/dacon3/ML_try_0202_2.pickle.dat', 'wb'))
# print('=====save complete=====')

# # 불러오기
# model = pickle.load(open('../data/xgb_save/dacon3/ML_try_0202_2.pickle.dat', 'rb'))
# print('======read complete=====')
#--------------------------------------------------------------------------------------------------------


# #4. 예측 + 저장
# sub = pd.read_csv('../data/csv/dacon3/submission.csv')
# sub['digit'] = model.predict(all_test)
# print(sub.head())

# # csv로 저장
# sub.to_csv('../data/csv/dacon3/sub_0202_2.csv', index = False)


# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# ==============================================
# ML_try_0202_1
# score:  0.5682926829268292 > dacon score: 0.5539215686
# ML_try_0202_2
# best_parameter:  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=1200, n_jobs=-1, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# score:  0.5829268292682926
