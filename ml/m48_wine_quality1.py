# x의 범위를 지정한 것 처럼 y이도 범위를 지정하여 라벨링을 바꿀 수 있지 않을까????
# 그 전에 머신러닝으로 돌려서 스코어를 확인해보자

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
import pandas_profiling

wine = pd.read_csv('../data/csv/winequality-white.csv',index_col=None, header=0, sep=';')
print(wine.head())
print(wine.shape)   # (4898, 12)
print(wine.describe())

wine_npy = wine.values

x = wine_npy[:, :11]
y = wine_npy[:, 11]

print(x.shape)
print(y.shape)
# (4898, 11)
# (4898,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=22)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape)  # (3918, 11) (980, 11)

# 모델 구성 머신러닝으로 하자!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier()
model = RandomForestClassifier()
# model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score: ', score)

# =================================
# 11로 했을 때 KNeighborsClassifier
# score:  0.5581632653061225

# 11로 했을 때 RandomForestClassifier
# score:  0.6938775510204082

# 11로 했을 때 XGBClassifier
# score:  0.6540816326530612

