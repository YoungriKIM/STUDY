from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)
# C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv
# 에가보면 파일이 csv 형태로 저장되어있음을 확인 할 수 있다.

# 파이썬에는 리스트와 딕셔너리가 매우 중요하다.
# {key : value}

# print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x_data = dataset.data
y_data = dataset.target
# 원래 이렇게 했는데 아래 방법도 가능하다.
# x = dataset['data']
# y = dataset['target']   #리스트에 들어간 키값 가체가 ''이라서 이렇게 작성해야 한다.
# print(x)
# print(y)

# 으아아아아아아아아아아아아ㅏㅏㅏ 나도 미치것어ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ

# print(dataset.frame)               #None
# print(dataset.target_names)        #['setosa' 'versicolor' 'virginica']
# print(dataset['DESCR'])            #이렇게 리스트안에 ''형태로 찍어도 된다.
# print(dataset['feature_names'])    #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(dataset.filename)            # C:\Users\Admin\anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv 

print(type(x_data), type(y_data))     #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

# np.save('../data/npy/iris_x_data.npy', arr=x_data)    # 저장 할 때는 npy로 한다.
# np.save('./data/npy/iris_y_data.npy', arr=y_data)    # data 폴더에 저장 된 것을 확인