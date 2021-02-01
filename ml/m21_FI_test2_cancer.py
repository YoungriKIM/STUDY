# feature_importances가 0인 컬럼들을 제거하여 데이터셋을 재구성 후
# DesisionTree로 모델을 돌려서 acc 확인!
# 해서 중요도가 0인 녀석들을 삭제하고 그래프 나오게 까지

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape) 

# 판다스의 데이터프레임 지정
df = pd.DataFrame(x, columns=dataset.feature_names)
df1 = df.iloc[:,[7,12,13,15,21,22,23,24]].values
names = dataset.feature_names[[7,12,13,15,21,22,23,24]]


# print(df2.info())
x_train, x_test, y_train, y_test = train_test_split(df1, dataset.target, train_size=0.8, random_state=311)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# (455, 8)
# (114, 8)
# (455,)
# (114,)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

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

#===========================================================
# feature_importances 와 시각화 / 0 제거 전
# [0.         0.         0.         0.         0.         0.
#  0.         0.74876667 0.         0.         0.         0.
#  0.00838369 0.01643413 0.         0.02738672 0.         0.
#  0.         0.         0.         0.05414488 0.08366905 0.05657739
#  0.00463748 0.         0.         0.         0.         0.        ]
# acc:  0.9385964912280702

#===========================================================
# feature_importances 와 시각화 / 0 제거 후
# [0.74876667 0.00838369 0.01643413 0.02738672 0.05414488 0.08366905
#  0.05657739 0.00463748]
# acc:  0.9385964912280702

