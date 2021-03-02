# 이전 파일의 그래프에서 점을 찍었던 최적의 w값을 찾자
# 이해를 위해 여러 변수를 바꿔서 찍어볼 것

# 러닝 레이트의 원리를 이해하자

import numpy as np

f = lambda x : x**2 - 4*x + 6

# 위 2차 함수를 미분한다.
gradient = lambda x : 2*x - 4

x0 = 10.0   # 처음은 랜덤한 값을 준다
epoch = 10
learning_rate = 00.3

print('step\tx\tf(x)')  # \t = 탭만큼 띄워준다
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))


# 그라디언트를 만들어보자
for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp   # 계속 조금씩 줄어들겠지?

    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))
# 결과치가 미분한 값인 2에 수렴하게 된다.

