# 0128과제
# train test 나눈 다음 train만 kfold 하여 val 만들지 말고
# train, test로 5등분 kfold한 다음 잘린 train 안에서 train_test_split으로 val도 만들기

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = SVC()

score = cross_val_score(model, x_train, y_train, cv = kfold)
print(score)

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

# ===================================================
# m10
# [1.         0.95833333 1.         0.95833333 0.91666667]

