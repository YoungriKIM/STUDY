# 데이터 전처리에 train_test_split / kfold / parameters 지정까지 했는데,
# 여기에 다른 전처리까지 더 더하는 것을 pipeline이라고 한다. 해보자

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline    # pipeline 과 make_pipeline 은 쓰는 방법이 다를 뿐 비슷한 녀석들이다.

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)         # 아래에 모델 지정할 때 믹맥스스케일러를 붙여줘서 필요없음!

#2. 모델
# model = Pipeline([('scaler', MinMaxScaler()), ('select', SVC())])         # 믹맥스스케일과 SVC라는 모델을 붙여준다.
# model = make_pipeline(MinMaxScaler(), SVC())                                # 이렇게 쓰는게 훨씬 간편하지만 이름이 겹칠까봐 위처럼 이름을 지정해줄 수도 있다.
model = make_pipeline(StandardScaler(), SVC()) 

model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print('결과치는', results, '입니다.')


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