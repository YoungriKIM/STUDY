# 데이터의 칼럼 중요도를 알아보자 feature_importances
# 그 중요도를 시각화해보자! 바 형태로

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=311)

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
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

#===========================================
# DNN 모델
# R2:  0.5189554519135346

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_leaf=10, min_samples_split=10)
# 최종정답률:  -0.19824315751288935
# 160.586200초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestRegressor(max_depth=6, min_samples_leaf=7, min_samples_split=5)
# 최종정답률:  -0.24412081536557184
# 8.216432초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.46103294644913173 입니다.
# Standard / 결과치는 0.4849888362806932 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.4792949742352127 입니다.

#===========================================================
# feature_importances 와 시각화
# [0.         0.         0.21077046 0.11702734 0.00298925 0.05317895
#  0.02319453 0.         0.55644047 0.03639899]
# acc:  0.42781481220401363