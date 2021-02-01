# 데이터의 칼럼 중요도를 알아보자 feature_importances
# 그 중요도를 시각화해보자! 바 형태로

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=311)

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
# feature_importances 와 시각화
# [0.         0.         0.         0.         0.         0.
#  0.         0.72209129 0.         0.         0.         0.
#  0.00838369 0.01643413 0.         0.01173717 0.         0.
#  0.         0.         0.         0.05414488 0.11034442 0.05657739
#  0.00463748 0.         0.         0.         0.         0.01564955]
# acc:  0.9649122807017544