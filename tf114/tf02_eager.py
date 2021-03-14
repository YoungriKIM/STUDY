# 가상환경 설치가 안 될 때 참고~!
# 베이스에서 텐서플로우 1점대를 사용하는 방법

# 즉시 실행 모드
# from tensorflow.python.framework.ops import disable_eager_execution 
import tensorflow as tf

print(tf.executing_eagerly())
# False # 지금 가상환경은 텐서플로1로 실행하고 있어서 True가 뜬다.

# 즉시 실행 모드를 diable 로 만들면 텐서플로1점대로 실행을 할 수 있다.
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())
# False

print(tf.__version__)

# ----------------------------------------------------
# hello world를 출력해보자
hello = tf.constant('Hello World')
print(hello)

# sess = tf.Session()   #텐서플로 1.13 까지
sess = tf.compat.v1.Session()   # 이렇게 써야 한다.
print(sess.run(hello))

# 베이스로 실행을 했을 때 !
# 'tesorflow' has no attribute 'Session'
# 텐서플로 2점대 부터는 세스런을 삭제해서 없다고 나온다.

# ===============================
# 베이스 일 때
# b'Hello World'