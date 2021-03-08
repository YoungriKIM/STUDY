# m46_outliers1 은 (10,) 이라서 가능했는데, 데이터가 행렬형태일때도 적용할 수 있도록 수정하자

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100], [100,20000,3,400,500,600,700,8,900,1000]])
aaa = aaa.transpose()
print(aaa.shape) # (10,2) 


print('------------------------------------------')

def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        print(i,'열의','1사분위(25%지점): ',  quartile_1)
        print(i,'열의','q2(50%지점): ',  q2)
        print(i,'열의','3사분위(75%지점): ',  quartile_3)
        iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
        print(i,'열의','iqr: ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)  # 하계
        upper_bound = quartile_3 + (iqr * 1.5)  # 상계
        print(i,'열의','lower_bound: ', lower_bound)
        print(i,'열의','upper_bound: ', upper_bound)

        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
        allout.append(a)

    return np.array(allout)


outlier_loc = outliers(aaa)
print('이상치의 위치:\n', outlier_loc)

# ---------------------------------------------------
# 0 열의 1사분위(25%지점):  3.25
# 0 열의 q2(50%지점):  6.5
# 0 열의 3사분위(75%지점):  97.5
# 0 열의 iqr:  94.25
# 0 열의 lower_bound:  -138.125
# 0 열의 upper_bound:  238.875
# 1 열의 1사분위(25%지점):  175.0
# 1 열의 q2(50%지점):  550.0
# 1 열의 3사분위(75%지점):  850.0
# 1 열의 iqr:  675.0
# 1 열의 lower_bound:  -837.5
# 1 열의 upper_bound:  1862.5
# 이상치의 위치:
#  [[array([4, 7], dtype=int64)]
#  [array([1], dtype=int64)]]
# ---------------------------------------------------
# boxplot 으로 이상치를 시각화해보자!
import matplotlib.pyplot as plt

plt.boxplot(aaa)    # 데이터만 넣어주면 된다.
plt.show()

