# feature_importances가 0인 컬럼들을 제거하여 데이터셋을 재구성 후
# DesisionTree로 모델을 돌려서 acc 확인!
# 해서 중요도가 0인 녀석들을 삭제하고 그래프 나오게 까지


# 0인 건 없는데 0번째 없애주겠음

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# 판다스의 데이터프레임 지정
df = pd.DataFrame(x, columns=dataset.feature_names)
df1 = df.iloc[:,[1,2,3]].values
names = dataset.feature_names[1:]


x_train, x_test, y_train, y_test = train_test_split(df1, dataset.target, train_size=0.8, random_state=311)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)       # 지금 정한 DecisionTree 라는 모델에 대한 중요도가 나온다.
print('acc: ', acc)

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

#===========================================================
# feature_importances 와 시각화 / 0 제거 후
# [0.03672022 0.38455518 0.5787246 ]
# acc:  0.9666666666666667