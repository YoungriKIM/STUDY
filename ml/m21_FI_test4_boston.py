# feature_importances가 0인 컬럼들을 제거하여 데이터셋을 재구성 후
# DesisionTree로 모델을 돌려서 acc 확인!
# 해서 중요도가 0인 녀석들을 삭제하고 그래프 나오게 까지

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_boston()
x = dataset.data
print(x.shape)

# 판다스의 데이터프레임 지정
df = pd.DataFrame(x, columns=dataset.feature_names)
df1 = df.iloc[:,[0,4,5,6,7,10,12]].values
names = dataset.feature_names[[0,4,5,6,7,10,12]]

# print(df2.info())
x_train, x_test, y_train, y_test = train_test_split(df1, dataset.target, train_size=0.8, random_state=311)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# (506, 13)
# (404, 7)
# (102, 7)
# (404,)
# (102,)

#2. 모델
model = DecisionTreeRegressor(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)       # 지금 정한 DecisionTree 라는 모델에 대한 중요도가 나온다.
print('acc: ', acc)



# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    plt.figure(figsize=(14,4))  
    n_features = df1.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), names)
    plt.xlabel('Feature Importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()


#===========================================
# DNN 모델
# R2:  0.8633197073667653

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_split=5)
# 최종정답률:  0.6734256301552818
# 30.139991초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestRegressor(n_jobs=-1)
# 최종정답률:  0.7078922745367211
# 8.782590초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.7287531921601408 입니다.
# Standard / 결과치는 0.7248061849054493 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.7396423200556946 입니다.

#===========================================================
# feature_importances 와 시각화
# [0.02841583 0.         0.         0.         0.00976672 0.67683071
#  0.00636    0.0691536  0.         0.         0.01774425 0.
#  0.1917289 ]
# acc:  0.4594683743312966

#===========================================================
# feature_importances 와 시각화 / 0 제거 후
# [0.02841583 0.02751097 0.67683071 0.00636    0.0691536  0.
#  0.1917289 ]
# acc:  0.4594683743312967