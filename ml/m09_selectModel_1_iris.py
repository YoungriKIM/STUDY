# ML으로 나온 값이 디폴트이다. 내가 딥러닝으로 구현한 모델이 ML보다 좋아야 한다.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators   # 이 안에는 많은 모델들이 들어있다.
from sklearn.datasets import load_iris 
import warnings

warnings.filterwarnings('ignore')       # 워닝 무시할거야 / 지우면 워닝이 모두 표시된다.

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

allAlgorithms = all_estimators(type_filter = 'classifier')
# all_estimators 안에 있는 것들 중 'classifier' 를 필터로 가져와서 쓰겠다.
# 즉 분류모델용만 쓰겠다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률:', accuracy_score(y_test, y_pred))
    except:
        # continue      # 아랫줄에 있는 프린트 대신 이 컨티뉴를 하면 에러가 있는 모델은 무시하고 끝까지 돌아간다.
        print(name, '은 없는 놈!')


# 오류  =======================================
# AdaBoostClassifier 의 정답들 0.9
# BaggingClassifier 의 정답들 0.9666666666666667
# BernoulliNB 의 정답들 0.26666666666666666
# CalibratedClassifierCV 의 정답들 0.9333333333333333
# CategoricalNB 의 정답들 0.9
# CheckingClassifier 의 정답들 0.3
# Traceback (most recent call last):
#   File "c:\Study\ml\m09_selectModel.py", line 19, in <module>
#     model = algorithm()
#   File "C:\Users\Admin\anaconda3\lib\site-packages\sklearn\utils\validation.py", line 72, in inner_f
#     return f(**kwargs)
# TypeError: __init__() missing 1 required positional argument: 'base_estimator'            

# 7번째에 나올 애한테 base_estimator를 지정 안 해줘서 멈췄다.
# 그래서 7번째 이후 모델을 볼 수 없으니, 이 에러를 무시하고 계속 갈 수는 없나?


# try지정 후 ================================                       # 에러가 있을 때 없다고 알려주고 계속 간다.
# AdaBoostClassifier 의 정답률: 0.9
# BaggingClassifier 의 정답률: 0.9666666666666667
# BernoulliNB 의 정답률: 0.26666666666666666     
# CalibratedClassifierCV 의 정답률: 0.9333333333333333
# CategoricalNB 의 정답률: 0.9
# CheckingClassifier 의 정답률: 0.3
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률: 0.7333333333333333
# DecisionTreeClassifier 의 정답률: 0.9666666666666667
# DummyClassifier 의 정답률: 0.5
# ExtraTreeClassifier 의 정답률: 0.9666666666666667   
# ExtraTreesClassifier 의 정답률: 0.9666666666666667
# GaussianNB 의 정답률: 1.0
# GaussianProcessClassifier 의 정답률: 0.9666666666666667
# GradientBoostingClassifier 의 정답률: 0.9666666666666667
# HistGradientBoostingClassifier 의 정답률: 0.9666666666666667
# KNeighborsClassifier 의 정답률: 0.9666666666666667
# LabelPropagation 의 정답률: 0.9666666666666667
# LabelSpreading 의 정답률: 0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률: 0.9666666666666667    
# LinearSVC 의 정답률: 0.9333333333333333
# LogisticRegression 의 정답률: 0.9666666666666667
# LogisticRegressionCV 의 정답률: 0.9666666666666667
# MLPClassifier 의 정답률: 0.9666666666666667
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률: 0.8666666666666667
# NearestCentroid 의 정답률: 0.9
# NuSVC 의 정답률: 0.9666666666666667
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률: 0.6333333333333333
# Perceptron 의 정답률: 0.9
# QuadraticDiscriminantAnalysis 의 정답률: 0.9666666666666667
# RadiusNeighborsClassifier 의 정답률: 0.9333333333333333
# RandomForestClassifier 의 정답률: 0.9666666666666667
# RidgeClassifier 의 정답률: 0.9
# RidgeClassifierCV 의 정답률: 0.9
# SGDClassifier 의 정답률: 0.7333333333333333
# SVC 의 정답률: 0.9666666666666667
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!


 
# 버전 확인하는 방법 ================================ 
# import sklearn
# print(sklearn.__version__)      # 0.23.2

#==============================================================================
#===========================================
# 딥러닝 모델
# acc:  0.9666666388511658

#===========================================
# ML 모델 중 최고
# GaussianNB 의 정답률: 1.0
