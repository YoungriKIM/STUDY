# machine 폴더 안의 test 폴더에 있는 car, tv 도 가져와서 써보자


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

print('---------------------------------')

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()
# =====================
# test_운전하다3
# test_시청하다3

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()
# =====================
# test_운전하다3
# test_시청하다3

from machine import test

test.car.drive()
test.tv.watch()
# =====================
# test_운전하다3
# test_시청하다3