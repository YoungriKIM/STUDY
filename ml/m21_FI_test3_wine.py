# feature_importances가 0인 컬럼들을 제거하여 데이터셋을 재구성 후
# DesisionTree로 모델을 돌려서 acc 확인!
# 해서 중요도가 0인 녀석들을 삭제하고 그래프 나오게 까지

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target
print(x.shape) 


# 판다스의 데이터프레임 지정
df = pd.DataFrame(x, columns=dataset.feature_names)
df1 = df.iloc[:,[0,1,6,9]].values
names = dataset.feature_names[0],dataset.feature_names[1],dataset.feature_names[6],dataset.feature_names[9]

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


#======================================================
# 딥러닝 모델
# acc:  0.9722222089767456

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=6)
# 최종정답률:  0.9722222222222222
# 48.837269초 걸렸습니다.

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  RandomForestClassifier(max_depth=8, min_samples_split=5)
# 최종정답률:  0.9722222222222222
# 9.028573초 걸렸습니다.

#===========================================================
# pipeline (스케일링 + 알고리즘)
# MinMax / 결과치는 0.9722222222222222 입니다.
# Standard / 결과치는 0.9722222222222222 입니다.

#===========================================================
# pipeline + GridSearchCV
# 결과치는 0.9444444444444444 입니다.

#===========================================================
# feature_importances 와 시각화
# [0.0337527  0.02893088 0.         0.         0.         0.
#  0.1825353  0.         0.         0.35920367 0.         0.
#  0.39557745]
# acc:  1.0

#===========================================================
# feature_importances 와 시각화 / 0 제거 후
# [0.11191396 0.03163755 0.45136689 0.4050816 ]
# acc:  0.9722222222222222