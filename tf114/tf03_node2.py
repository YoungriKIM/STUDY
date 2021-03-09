# 실습
# 덧셈/ 뺄셈/ 곱셈/ 나눗셈/ 나머지 만들어라

import tensorflow as tf
sess = tf.Session()

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(3.0, tf.float32)

node3 = tf.add(node1, node2)
print('add 덧셈: ', sess.run(node3))

node4 = tf.subtract(node1, node2)
print('sub 뺄셈: ', sess.run(node4))

node5 = tf.multiply(node1, node2)
print('mul 곱셈: ', sess.run(node5))

node6 = tf.divide(node1, node2)
print('divide 나눗셈: ', sess.run(node6))

node7 = tf.math.mod(node1, node2)
print('math.mod 나머지: ', sess.run(node7))

# add 덧셈:  5.0
# sub 뺄셈:  -1.0
# mul 곱셈:  6.0
# divide 나눗셈:  0.6666667
# math.mod 나머지:  2.0

# --------------------------------
# TensorFlow 연산   축약 연산자   설명
# tf.add()   a + b   a와 b를 더함
# tf.multiply()   a * b   a와 b를 곱함
# tf.subtract()   a - b   a에서 b를 뺌
# tf.divide()   a / b   a를 b로 나눔
# tf.pow()   a ** b     를 계산
# tf.mod()   a % b   a를 b로 나눈 나머지를 구함
# tf.logical_and()   a & b   a와 b의 논리곱을 구함. dtype은 반드시 tf.bool이어야 함
# tf.greater()   a > b     의 True/False 값을 반환
# tf.greater_equal()   a >= b     의 True/False 값을 반환
# tf.less_equal()   a <= b     의 True/False 값을 반환
# tf.less()   a < b     의 True/False 값을 반환
# tf.negative()   -a   a의 반대 부호 값을 반환
# tf.logical_not()   ~a   a의 반대의 참거짓을 반환. tf.bool 텐서만 적용 가능
# tf.abs()   abs(a)   a의 각 원소의 절대값을 반환
# tf.logical_or()   a I b   a와 b의 논리합을 구함. dtype은 반드시 tf.bool이어야 함