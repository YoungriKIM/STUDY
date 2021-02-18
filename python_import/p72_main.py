# 변수도 불러와서 써보자


import p71_byunsu as p71


print(p71.aaa)
print(p71.square(10))
#======================
# 2
# 1024


print('===========================')


from p71_byunsu import aaa, square

print(aaa)
#======================
# 2

print('===========================')


from p71_byunsu import aaa, square

aaa = 3

print(aaa)
print(square(10))
# ===========================
# 3
# 1024      # 역시 2의 10승이 나왔다. p71 안에 aaa=2로 정의해놨기 때문이다.
