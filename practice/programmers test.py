#===========================================================
# 전화번호 뒤에 4자리 뺴고 별으로

phone = '01076591799'

print(len(phone))

def solution(phone):
    return '#'*(len(phone)-4) + phone[-4:]

solution(phone)

# 리스트로
phone2 = ['01012345678', '01034567890', '01024683579', '023456421', '0314459231']

# print(len(phone2[1]))
def solution2(phone):
    x = []
    for i in range(0,5):
        tmp =  '*'*(len(phone2[i])-4) + phone2[i][-4:]
        x.append(tmp)
    return(x)

solution2(phone2)


#===========================================================
# 겹치는 스트링 삭제
list1 = ['aa','bb','bb','cc','dd']
list2 = ['aa','bb']

import collections

list3 = collections.Counter(list1)-collections.Counter(list2)
print(list(list3)) #리스트로 묶어서 내보내야 함


#===========================================================
# 원하는 만큼 반복하고 자르기
su = '수박'

def solution(string, n):
    result = []
    result = string*4
    return result[:n]

print(solution(su, 3))


#===========================================================

