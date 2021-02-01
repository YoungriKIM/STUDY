# 데이터의 칼럼 중요도를 알아보자 feature_importances
# 그 중요도를 시각화해보자! 바 형태로

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=311)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)       # 지금 정한 DecisionTree 라는 모델에 대한 중요도가 나온다.
print('acc: ', acc)
# =================================================================
# [0.         0.03672022 0.38455518 0.5787246 ]     > 첫번째, 두번째 칼럼은 빼고 해도 될것 같다는 결론
# acc:  0.9666666666666667

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
# 딥러닝 모델
# acc:  0.9666666388511658

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667
# 22.954975초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667
# 7.167455초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.9666666666666667 입니다.
# Standard / 결과치는 0.9666666666666667 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.9666666666666667 입니다.

#===========================================================
# feature_importances 와 시각화 / 0 제거 전
# [0.01669101 0.02002921 0.89206482 0.07121497]
# acc:  0.9666666666666667