# all_estimator + KFold  가자!

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators   # 이 안에는 많은 모델들이 들어있다.
from sklearn.datasets import load_iris 
import warnings

warnings.filterwarnings('ignore')       # 워닝 무시할거야 / 지우면 워닝이 모두 표시된다.

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

kfold = KFold(n_splits=5, shuffle=True) # 행만 섞는다는 거 기억~


allAlgorithms = all_estimators(type_filter = 'classifier')
# all_estimators 안에 있는 것들 중 'classifier' 를 필터로 가져와서 쓰겠다.
# 즉 분류모델용만 쓰겠다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # score = cross_val_score(model, x_train, y_train, cv=5) 도 되는데 대신 shuffle은 안된다.
        # model.fit(x_train, y_train) 핏은 필요 없으니까
        # y_pred = model.predict(x_test)
        # print(name,'의 정답률:', accuracy_score(y_test, y_pred))  크로스발이 알아서 애큐러시 스코어를 반환하니까
        print(name, '의 정답률: \n', scores)
    except:
        # continue      # 아랫줄에 있는 프린트 대신 이 컨티뉴를 하면 에러가 있는 모델은 무시하고 끝까지 돌아간다.
        print(name, '은 없는 놈!')


#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

#===========================================
# ML 모델 중 최고
# GaussianNB 의 정답률: 1.0

# kfold적용 후 모든 모델에 대한 정답률이 5개씩 나온다.=========================
# LogisticRegressionCV 의 정답률:
# [0.91666667 0.875      1.         1.         1.        ]
