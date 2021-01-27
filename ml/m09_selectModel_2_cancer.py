from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률', accuracy_score(y_test, y_pred))
    except:
        continue
        # print(name, '은 없다!')


#===========================================
# 딥러닝 모델
# acc: 0.9736841917037964

#===========================================
# ML 모델 중 최고
# AdaBoostClassifier 의 정답률 0.9736842105263158