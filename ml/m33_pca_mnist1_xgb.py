# m31로 만든 0.95 n_component를 사용하여
# XGB(디폴트) 모델을 만들 것

# xgb 디폴트의 성능이 크게 좋지 않은데, 파라미터 튜닝이 없기 때문이라고 추측할 수 있다.
# m34에는 파라미너 튜닝을 해보자


import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. x 데이터 불러오고 pca적용 ===================================
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
# print(x.shape)      #(70000, 28, 28) / x_train = 60000,28,28 / x_test = 10000,28,28
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)      #(70000, 784)

# pca 적용
pca = PCA(n_components = 154)
x2 = pca.fit_transform(x)
# print(x2.shape)            # (70000, 154)
# pca_EVR = pca.explained_variance_ratio_ # 변화율
# print('pca_EVR: ', pca_EVR)
# print('cumsum: ', sum(pca_EVR))    # cumsum: 0.9500035195029432

# train_test_split
x_train = x2[:60000, :]
x_test = x2[60000:, :]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)
# (60000, 154)
# (10000, 154)

#1. y 데이터 불러오고 전처리  ===================================
(_, y_train), (_, y_test) = mnist.load_data()

#2. 모델
model = XGBClassifier(n_jobs=-1, use_label_encoder=False, n_estimator= 2000)  
# n_estimator : 나무의 수, epoch와 비슷하게 생각

#3. 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train, verbose = True, eval_metric='mlogloss', eval_set=[(x_train, y_train), (x_test, y_test)])

#4. 평가(스코어)
score = model.score(x_test, y_test)
print('score: ', score)

# ==============================================================================
# 40-2 mnist CNN
# loss, acc:  0.009633197449147701 0.9853999743461609     137

# 40-3 mnist DNN       
# loss:  [0.10886456072330475, 0.9815000295639038]
# y_pred:  [7 2 1 0 4 1 4 9 6 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]

# m32_1 pca 0.95이상으로 지정한 파일
# loss:  [0.11894247680902481, 0.9639000296592712]

# m33_1 pca 0.95이상으로 지정한 파일 > XGBoost
# score:  0.9634