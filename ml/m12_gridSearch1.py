# m10-1을 가져와서 씀

# 원하는 것만 쓰고 그것도 하이퍼파라미터 튜닝해보자

import numpy as np
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
# GridSearchCV 채로 걸러서 다 찾은다음 지정해줄 거야. cross_validation처럼, 그래서 cross_val_score가 필요가 없다.
from sklearn.metrics import accuracy_score

# 사이킷런에서 제공하는 아래에 있는 모델들을 불러와서 실행하고 비교해보자
from sklearn.svm import LinearSVC, SVC
import warnings
warnings.filterwarnings('ignore')

#1. 데이터 불러오기
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

kfold = KFold(n_splits=5, shuffle=True) 

parameters = [
    {'C' : [1,10,100,1000], 'kernel' : ['linear']},                                 # 4번
    {'C' : [1,10,100], 'kernel':['rdf'], 'gamma':[0.001, 0.0001]},                  # 6번
    {'C' : [1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}          # 8번   > 18번을 5번 > 90번
]

#2. 모델 구성
# model = SVC()
model = GridSearchCV(SVC(), parameters, cv = kfold)
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

# MinMaxScaler 일 때 ====================================
# model = LinearSVC()
# result:  0.9666666666666667
# accuracy_score:  0.75

# model = SVC()
# result:  1.0
# accuracy_score:  1.0

# model = KNeighborsClassifier()
# result:  1.0
# accuracy_score:  1.0

# model = LogisticRegression()
# result:  1.0
# accuracy_score:  1.0

# model = DecisionTreeClassifier()
# result:  0.9333333333333333
# accuracy_score:  1.0

# model = RandomForestClassifier()
# result:  0.9333333333333333
# accuracy_score:  0.75
