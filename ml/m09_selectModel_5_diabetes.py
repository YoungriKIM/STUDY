from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=311)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))
    except:
        continue
        # print(name, '은 없어 돌아가')

#===========================================
# 딥러닝 모델
# R2:  0.5189554519135346

#===========================================
# ML 모델 중 최고
# LassoLarsIC 의 정답률:  0.511433247804918