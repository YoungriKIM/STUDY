# 이름 보려고 만들었던 파일을 불러와보자

import p11_car
# p11_car.py의 module 이름은:  p11_car
# 이름이 __main__이 아니라 불러온 것은 불러온 파일의 파일명이 나온다.

import p12_tv
# p12_tv.py의 module 이름은:  p12_tv


print('----------------------------')
print('p13_do.py의 module 이름은: ', __name__)
print('----------------------------')


p11_car.drive()
p12_tv.watch()
# 운전하다
# 시청하다

# import 에서 함수 한번에 가져온 뒤에 이 저 줄은 그 함수 안에 정의한 drive, watch만 불러오겠다는 뜻이다.