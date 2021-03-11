# 실습: sess.run() / InteractiveSession / eval 사용해서 하이포시스 출력해라
# 이전 파일 참고
import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b
# --------------------------------------------
# print('hypothesis: ', )
# --------------------------------------------

# 1) sess.run 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print('첫 번째 방법: ', aaa) 
sess.close()

# 2) InteractiveSession 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval()
print('두 번째 방법: ', bbb)
sess.close()

# 3) .eval(session=sess) 통과해서 출력 ----------------------------------------------
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print('세 번째 방법: ', ccc)

# 첫 번째 방법:  [1.3       1.6       1.9000001]
# 두 번째 방법:  [1.3       1.6       1.9000001]
# 세 번째 방법:  [1.3       1.6       1.9000001]