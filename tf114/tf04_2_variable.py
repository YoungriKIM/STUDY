# variable 을 알아보자

import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

# 이 variation 자료형은 세션을 통과하기 전에 아래처럼 초기화를 시키는 것이 정해진 문법이다.
init = tf.global_variables_initializer()

sess.run(init)

print(sess.run(x))
# [2.]

# 그런데 에러가 뜬다. 지금 쓰고 있는 문법이 지금 버전의 이전 것이라 그렇다. 
# Please use tf.compat.v1.global_variables_initializer instead.

