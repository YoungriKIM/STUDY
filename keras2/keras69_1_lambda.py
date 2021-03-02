# 람다를 이용해 간단한 함수를 만들어 보자

gradient = lambda x: 2*x - 4
#위 코드는 아래와 같다. 

def gradient2(x) :
    temp = 2*x - 4
    return temp

x = 3
print(gradient(x))
print(gradient2(x))

# 2
# 2

# 둘의 결과는 같다.