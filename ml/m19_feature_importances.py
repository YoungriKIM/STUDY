# 데이터의 칼럼 중요도를 알아보자 feature_importances

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=311)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)       # 지금 정한 DecisionTree 라는 모델에 대한 중요도가 나온다.
print('acc: ', acc)

# =================================================================
# [0.         0.03672022 0.38455518 0.5787246 ]     > 첫번째, 두번째 칼럼은 빼고 해도 될것 같다는 결론
# acc:  0.9666666666666667