# 클래스도 가능하니까 해보자

# 클래스는 __init__ 이 필요하다.
# 클래스는 이것저것 여러가지를 넣을 수 있다.
# 파이썬의 클래스에는 self가 있는데, 이는 클래스 자체를 의미한다.

class person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address
    
    def greeting(self):    # 클래스 안에 들어가는 함수에는 반드시 (self) 나 자신이 들어가야 한다.
                           # slef를 넣지 않고 불러와서 쓰면 TypeError: greeting() takes 0 positional arguments but 1 was given 이런 에러가 뜬다.
        print('안녕하세요, 저는 {0}입니다.'.format(self.name))

# 자 {0}의 뜻은 format에 들어가 있는 것의 0번째를 넣는다는 뜻이다. 