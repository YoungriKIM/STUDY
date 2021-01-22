# 뒤의 네자리 빼고 *표시로 바꾸기

#=======================================================
# phone = '01076591799'

# print(len(phone))

# def solution(phone):
#     return '#'*(len(phone)-4) + phone[-4:]

# solution(phone)

#=======================================================

phone2 = ['01012345678', '01034567890', '01024683579', '023456421', '0314459231']

# print(len(phone2[1]))
def solution2(phone):
    x = []
    for i in range(0,5):
        tmp =  '*'*(len(phone2[i])-4) + phone2[i][-4:]
        x.append(tmp)
    return(x)

solution2(phone2)