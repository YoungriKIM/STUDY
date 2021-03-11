# 출력 할 때 꼭 sess.run을 통과해야 하는 것은 아니다. 
# 다른 방법도 알아보자

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')

print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
# 세스런 통과 안 했으니 그냥 자료형 구조가 나온다.

# 1) sess.run 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print('aaa: ', aaa)  # [2.2086694]
sess.close()

# 2) InteractiveSession 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
print('bbb: ', bbb)
sess.close()

# 3) .eval(session=sess) 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)
print('ccc: ', ccc)

# aaa:  [2.2086694]
# bbb:  [2.2086694]
# ccc:  [2.2086694]