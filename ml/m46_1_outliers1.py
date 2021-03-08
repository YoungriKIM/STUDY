# 이상치도 처리해보자!
# outlier: 이상치
# 1. 0으로 처리
# 2. 1,2,3,4,10만 > 10만을 nan으로 바꾼 후 보간법
# 등등 

# 참고 자료
# https://m.blog.naver.com/lingua/221909198917

import numpy as np

aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])
# 데이터의 갯수가 10개로 짝수이다. 홀수일 때의 계산법은 또 다르다.

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])      # percentile: 사분위, 어느 지점 / 25: 25% 지점, 50: 50%지점 ...
    print('1사분위(25%지점): ',  quartile_1)
    print('q2(50%지점): ',  q2)
    print('3사분위(75%지점): ',  quartile_3)
    iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
    print('iqr: ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)  # 하계
    upper_bound = quartile_3 + (iqr * 1.5)  # 상계
    print('lower_bound: ', lower_bound)
    print('upper_bound: ', upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | : 엔터 위에 있는 키 !

# 1사분위(25%지점):  3.25
# q2(50%지점):  6.5
# 3사분위(75%지점):  97.5
# iqr:  94.25
# lower_bound:  -138.125
# upper_bound:  238.875
# >>> 정상적인 값의 범위가 -138~238 사이라고 평가한다. 즉 5000, 10000은 이상치로 분류한다.
# 평균값을 이용했다면 이런 결과가 나올 수가 없다.

# 왜 1.5를 곱해주나요? # 임의로 해준 것이다.

outlier_loc = outliers(aaa)
print('이상치의 위치: ', outlier_loc)
# 이상치의 위치:  (array([8, 9], dtype=int64),)

# ---------------------------------------------------
# 이상치를 처음에 넣어보자
bbb = np.array([10000, 1,2,3,4,6,7,90,100,5000])
print('bbb 이상치의 위치: ', outliers(bbb))
# 1사분위(25%지점):  3.25
# q2(50%지점):  6.5
# 3사분위(75%지점):  97.5
# iqr:  94.25
# lower_bound:  -138.125
# upper_bound:  238.875
# bbb 이상치의 위치:  (array([0, 9], dtype=int64),)

# 이상치를 중간에 넣어보자
ccc = np.array([1,2,3,4,6, 10000, 7,90,100,5000])
print('ccc 이상치의 위치: ', outliers(ccc))
# 1사분위(25%지점):  3.25
# q2(50%지점):  6.5
# 3사분위(75%지점):  97.5
# iqr:  94.25
# lower_bound:  -138.125
# upper_bound:  238.875
# ccc 이상치의 위치:  (array([5, 9], dtype=int64),)

## >> 이상치의 위치만 바뀔 뿐 계산되는 값이 바뀌지 않는다.

# ---------------------------------------------------
# boxplot 으로 이상치를 시각화해보자!
import matplotlib.pyplot as plt

plt.boxplot(aaa)    # 데이터만 넣어주면 된다.
plt.show()

# 1사분위(25%지점):  3.25 를 박스 하단
# 3사분위(75%지점):  97.5 를 박스 상단으로 하여 그려준다.
