# gridSearch 의 단점: 모든 경우의 수를 다 해서 시간이 오래 걸린다
# randomSearch 의 장점: 랜덤으로 잡아서 하니까 시간이 적게 걸리고, 알아서 파라미터를 잡아주니 덜 감성적인 코딩이 된다.

# 그리드 서치를 랜덤 서치로 바꿔보자

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
# CV가 붙은 건 corss_validation이 붙어 있다는 뜻!
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
# 까먹을까봐 판다스 csv 불러오기 함 하자
import pandas as pd
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x =  dataset.iloc[:,:-1]
y =  dataset.iloc[:,-1]

# dataset = load_iris()
# x = dataset.data
# y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

kfold = KFold(n_splits=5, shuffle=True) 

# 랜덤이어도 파라미터는 내가 정해주어야 한다. 넣지 않은 파라미터는 알아서 디폴트값으로 넣는다.
parameters = [
    {'C' : [1,10,100,1000], 'kernel' : ['linear']},                                 # 4번
    {'C' : [1,10,100], 'kernel':['rdf'], 'gamma':[0.001, 0.0001]},                  # 6번
    {'C' : [1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}          # 8번   > 18번을 5번 > 90번
]

#2. 모델 구성
# model = SVC()
model = RandomizedSearchCV(SVC(), parameters, cv = kfold)
# parameters 안의 것들은 SVC 안에 들어있던 것임

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수: ', model.best_estimator_)
# model.best_estimator_ : 90번 돈 것중 가장 최고를 알려준다.

# 이렇게 프레딕트를 추가하면 90번중 가장 좋은 것을 알아서 지정해준다.
y_pred = model.predict(x_test)
print('최종정답률 :', accuracy_score(y_pred, y_test))
# print('최종정답률 :', model.score(x_test, y_test)) 도 가능하고 윗줄과 이줄은 같다.


#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

# gridsearchCV 일 때 ====================================
# 최적의 매개변수:  RandomForestClassifier(max_depth=10, min_samples_split=3)
# 최종정답률:  0.9666666666666667

# RandomizedSearchCV 적용 =============================
# 최적의 매개변수:  SVC(C=1, kernel='linear')
# 최종정답률 : 0.9666666666666667