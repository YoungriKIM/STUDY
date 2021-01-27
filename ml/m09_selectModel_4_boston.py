
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators   # 이 안에는 많은 모델들이 들어있다.
from sklearn.datasets import load_iris, load_boston
import warnings

warnings.filterwarnings('ignore')       # 워닝 무시할거야

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

allAlgorithms = all_estimators(type_filter = 'regressor')
#  all_estimators 안에 있는 것들 중 'classifier' 를 필터로 가져와서 쓰겠다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률:', r2_score(y_test, y_pred))      # 회귀는 r2_score 써야함 주의!
    except:
        # continue      # 아랫줄에 있는 프린트 대신 이 컨티뉴를 하면 에러가 있는 모델은 무시하고 끝까지 돌아간다.
        print(name, '은 없는 놈!')


#===========================================
# 딥러닝 모델
# R2:  0.8633197073667653

#===========================================
# ML 모델 중 최고
# ExtraTreesRegressor 의 정답률: 0.8229304527912591