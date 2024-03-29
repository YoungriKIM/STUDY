# 데이터 전처리에 train_test_split / kfold / parameters 지정까지 했는데,
# 여기에 다른 전처리까지 더 더하는 것을 pipeline이라고 한다. 해보자

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline    # pipeline 과 make_pipeline 은 쓰는 방법이 다를 뿐 비슷한 녀석들이다.

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)         # 아래에 모델 지정할 때 믹맥스스케일러를 붙여줘서 필요없음!

#2. 모델
# model = Pipeline([('scaler', MinMaxScaler()), ('select', RandomForestClassifier())])         # 믹맥스스케일과 SVC라는 모델을 붙여준다.
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())                                # 이렇게 쓰는게 훨씬 간편하지만 이름이 겹칠까봐 위처럼 이름을 지정해줄 수도 있다.
model = make_pipeline(StandardScaler(), RandomForestClassifier())                             

model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print('결과치는', results, '입니다.')


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