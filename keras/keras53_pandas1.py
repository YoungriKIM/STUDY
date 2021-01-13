import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values())     #키가 있으면 밸류가 있겠지

print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

# 판다스에 데이터를 넣을 것인데, 행렬이 필요하고. 이 아이리스에는 data가 행렬로 되어있다.

x = dataset.data
# x = dataset['data']       >  이것도 가능하다.
y = dataset.target
# y = dataset['target']     >  이것도 역시 가능

print(x)
print(y)
print(x.shape, y.shape)     #(150, 4) (150,)
print(type(x), type(y))     #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

# dataset.feature_names = 칼럼명, 판다스의 헤더명

df = pd.DataFrame(x, columns=dataset.feature_names)
# df = pd.DataFrame(x, columns=dataset['feature_names'])      > 이것도 가능
print(df)   # 표처럼 정리되어 보인다
print(df.shape)     # 참고로 리스트는 쉐잎이 나오지 않는다. 그래서 우리가 지금까지 리스트를 np.array로 해준 것이다.
print(df.columns)      # 헤더부분이 명시된다.
print(df.index)     #RangeIndex(start=0, stop=150, step=1)      > 0부터 149까지 150개의 인덱스가 있다. (행으로 이해하면 쉬움)
# 그런데 위에서 df의 칼럼은 지정해주었는데 인덱스는 해주지 않았다. 인덱스는 지정하지 않아도 0부터 지정이 된다.


print(df.head())        # = df[:5]  이 데이터의 위에서부터 5개를 잘라서 미리 보여준다. 데이터 파악하라고
print(df.tail())        # = df[-5:] 이 데이터의 아래에서부터 5개를 보여준다.
print(df.info())
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 4 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
# dtypes: float64(4)
# memory usage: 4.8 KB
# None

print(df.describe())            # numpy랑 다르게 이렇게 한번에 볼 수 있다.
#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238       >표준 편차
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

# 칼럼명이 너무 기니까 수정해보자
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']        #이렇게 하면 칼럼명이 갱신이 된다
print(df.columns)       # Index(['sepal lenght', 'sepal width', 'petal length', 'pretal width'], dtype='object')
print(df.info())
print(df.describe())    # > 세가지 모두 바뀌었음을 확인


# 지금까지는 df에 x만 넣었고 여기에 y를 붙여주자
print(df['sepal_length'])       #sepal_length 열의 정보만 나온다. 이렇게 새로운 열을 추가하고싶다.

df['Target'] = dataset.target # = y
print(df.head())

print(df.shape)   #(150, 5)     > y가 추가 되어서
print(df.columns)       #Index(['sepal_length', 'sepal_width', 'petal_length', 'pretal_width', 'Target']
print(df.index)         #RangeIndex(start=0, stop=150, step=1) 로 동일하다.
print(df.tail())

print(df.info())
print(df.isnull())             # 결과가 전부 False로 나오는데 결측치가 없다는 뜻이다.
print(df.isnull().sum())       # 결측치의 수를 보여준다.
print(df.describe())

print(df['Target'].value_counts())
# 2    50
# 1    50
# 0    50
# target 안에 0이 50개, 1이 50개, 2가 50개

# 상관계수를 구해보자       #상관계수correlation coefficien / 칼럼간의 상관계수를 볼 것이며 / 우리에게 가장 중요한 feature(칼럼)는 target(y)이다.
print(df.corr())
#               sepal_length  sepal_width  petal_length  pretal_width    Target
# sepal_length      1.000000    -0.117570      0.871754      0.817941  0.782561
# sepal_width      -0.117570     1.000000     -0.428440     -0.366126 -0.426658
# petal_length      0.871754    -0.428440      1.000000      0.962865  0.949035
# pretal_width      0.817941    -0.366126      0.962865      1.000000  0.956547
# Target            0.782561    -0.426658      0.949035      0.956547  1.000000


# 표를 보기 좋게해보자 > 히트맵으로
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)     #데이터는 df의 상관계수로 사각형으로 안에 글씨를 넣어서 cbar도 추가해서
plt.show()

# 도수 분포도
plt.figure(figsize=(10, 8))        #(10,6)짜리 도화지를 준비했고

plt.subplot(2, 2, 1)
plt.hist(x = 'sepal_length', data=df) # 여기서 말하는 hist는 도수 분포표임 / 도수분포도: histogram 
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x = 'sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x = 'petal_length', data=df)
plt.title('petal_length')

plt.subplot(2, 2, 4)
plt.hist(x = 'petal_width', data=df)
plt.title('petal_width')

plt.show()
