# GridSearch / Randomized 적용해서 cnn,dnn보다 잘 나오게 튜닝

# 제공된 파라미터
# parameters = [
#     {'n_estimators' : [100,200], 'learning_rate' : [0.1, 0.3, 0.001], 'max_depth' : [4,5,6]},
#     {'colsample': [0.6, 0.9, 1], 'learning_rate' : [0.5, 0.01, 0.2], 'colsample_bytree': [0.5, 2, 0.1]},
#     {'n_estimators' : [50,110], 'learning_rate' : [1, 0.2, 0.005], 'max_depth' : [2,4,8], 'colsample_bytree': [0.6, 0.9, 1]},
# ]

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. x 데이터 불러오고 pca적용 ===================================
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
# print(x.shape)      #(70000, 28, 28) / x_train = 60000,28,28 / x_test = 10000,28,28
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)      #(70000, 784)

# pca 적용
pca = PCA(n_components = 713)
x2 = pca.fit_transform(x)
# print(x2.shape)            # (70000, 713)
# pca_EVR = pca.explained_variance_ratio_ # 변화율
# print('pca_EVR: ', pca_EVR)
# print('cumsum: ', sum(pca_EVR))    # cumsum: 0.9500035195029432

# train_test_split
x_train = x2[:60000, :]
x_test = x2[60000:, :]

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)
# (60000, 713)
# (10000, 713)

#1. y 데이터 불러오고 전처리  ===================================
(_, y_train), (_, y_test) = mnist.load_data()

#2. 모델
parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1,2,4]}
]

model = GridSearchCV(RandomForestClassifier(), parameters)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수: ', model.best_estimator_)
score = model.score(x_test, y_test)
print('모델 스코어: ',score)

y_pred = model.predict(x_test)
print('애큐러시 스코어: ', accuracy_score(y_pred, y_test))


# ==============================================================================
# 40-2 mnist CNN
# loss, acc:  0.009633197449147701 0.9853999743461609     137

# 40-3 mnist DNN       
# loss:  [0.10886456072330475, 0.9815000295639038]
# y_pred:  [7 2 1 0 4 1 4 9 6 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]

# m32_1 pca 0.95이상으로 지정한 파일
# loss:  [0.2978833317756653, 0.9678000211715698]

# m32_2 pca 1.0 이상으로 지정한 파일
# loss:  [0.2978833317756653, 0.9678000211715698]

# m34_2 pca 1.0 이상으로 지정한 파일 > XGBoost < GridSearch
# 모델 스코어:  0.9156

# m34_2 pca 1.0이상으로 지정한 파일 > XGBoost < RandomizedSearch
# 모델 스코어:  0.9125