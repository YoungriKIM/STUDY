# machine 폴더 안에 있는 것들을 불러오자
# + from 으로 써보자

from machine.car import drive
from machine.tv import watch

drive()
watch()
# =======================
# 운전하다2
# 시청하다2

print('---------------------------------')

from machine import car
from machine import tv

car.drive()
tv.watch()
# =====================
# 운전하다2
# 시청하다2

print('---------------------------------')

from machine import car, tv

car.drive()
tv.watch()
# =====================
# 운전하다2
# 시청하다2

