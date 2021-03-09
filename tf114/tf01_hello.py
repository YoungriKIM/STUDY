# TensorFlow Snippets 확장에서 검색 후 다운해야 자동완성 가능

# 텐서플로 2점대로 바뀐 지 1년 정도밖에 안되었기 때문에 실무에 가면 1점대를 사용할 일이 분명 있다는 점

# 가상환경에 텐서플로우 버전 1.14.0 가 잘 설치되었는지 보자
# 비주얼코드 왼쪽 아래에서 경로를 tf114로 바꿔서 비교 확인

import tensorflow as tf
print(tf.__version__)   

# base 환경
# 2.3.1

# tf114 환경
# 1.14.0

# ----------------------------------------------------
# hello world를 출력해보자
hello = tf.constant('Hello World')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 자료형의 구조만 나온다. 모든 것은 세션을 통과해야 출력된다. 왜 귀찮게 세션 통과하는지는 tf3 참고

sess = tf.Session()
print(sess.run(hello))
# b'Hello World'
