# 머신러닝으로 디폴트 잡으려고 츄라이 함

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 지정
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

x = train.drop(['id','digit','letter'], axis=1).values
y = train['digit']

# y = train['digit']
# y2 = np.zeros((len(y), len(y.unique())))
# for i , digit in enumerate(y):
#     y2[i, digit] = 1

# print(x.shape)      #(2048, 784)
# print(y.shape)      #(2048, 10)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=519)

# 파라미터 지정
parameters = [
    {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
    {'colsample': [0.6, 0.9, 1], 'colsample_bytree': [0.5, 2, 0.1]},
    {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'colsample_bytree': [0.6, 0.9,1]},
]

kfold = KFold(n_splits=5, shuffle=True, random_state=311)

all_test = test.drop(['id', 'letter'], axis=1).values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

#--------------------------------------------------------------------------------------------------------
# #2. 모델
# model = RandomizedSearchCV(XGBClassifier(n_jobs=-1), parameters, cv = kfold)

# #3. 컴파일ㄴ 훈련ㅇ
# model.fit(x_train, y_train, verbose=True, eval_metric = ['mlogloss'], eval_set=[(x_train, y_train), (x_test, y_test)])

# #4. 평가
# score = model.score(x_test, y_test)
# print('score: ', score)

# --------------------------------------------------------------------------------------------------------
# 모델 pickle로 저장
# pickle.dump(model, open('../data/xgb_save/dacon3/ML_try_0202_1.pickle.dat', 'wb'))
# print('=====save complete=====')

# 불러오기
model = pickle.load(open('../data/xgb_save/dacon3/ML_try_0202_1.pickle.dat', 'rb'))
print('======read complete=====')
#--------------------------------------------------------------------------------------------------------


#4. 예측 + 저장
sub = pd.read_csv('../data/csv/dacon3/submission.csv')
sub['digit'] = model.predict(all_test)
print(sub.head())

# csv로 저장
sub.to_csv('../data/csv/dacon3/sub_0202_1.csv', index = False)


# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# ==============================================
# ML_try_0202_1
# score:  0.5682926829268292 > dacon score: 0.5539215686


