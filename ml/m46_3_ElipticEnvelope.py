# m46의 기능이 이미 만들어져있겠지?
# EllipticEnvelope 기능을 불러와서 이상치의 위치를 찾아보자
# https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html

from sklearn.covariance import EllipticEnvelope
import numpy as np


# ===================================
# 일차원으로 넣어보자!

aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100, 5000]])
aaa = np.transpose(aaa)
print(aaa.shape)
# (11, 1)

outlier = EllipticEnvelope(contamination=.3)
# contamination: 오염 / .2 : 20%의 이상치를 찾아라
outlier.fit(aaa)

print(outlier.predict(aaa))

# ===================================
# contamination .2
# [ 1  1 -1  1  1  1  1  1  1  1  1]

# contamination .1
# [ 1  1 -1  1  1  1  1  1  1  1  1]
# 10%인 0.1이 contamination의 디폴트이다.

# contamination .3
# [ 1  1 -1  1  1  1  1  1  1 -1 -1]
# 데이터의 30% 부분이라는 것은 이상치가 아닐 가능성이 높다.




# ===================================
# 다차원도 넣어보자!
bbb = np.array([[1,2,3,4,10000,6,7,5000,90,100], [100,20000,3,400,500,600,700,8,900,1000]])
bbb = np.transpose(bbb)
print(bbb.shape)
# (10, 2)

outlier2 = EllipticEnvelope(contamination=.2)
# contamination: 오염 / .2 : 20%의 이상치를 찾아라
outlier2.fit(bbb)

print(outlier2.predict(bbb))
# (10, 2)
# [ 1  1  1  1 -1  1  1 -1  1  1]

## !! 이 기능에서 행은 연대로 간다! 2개의 열을 따로 하고 싶다면 나눠서 해야 한다. 그럴 때는 for문이나 함수 만들어서 하자.