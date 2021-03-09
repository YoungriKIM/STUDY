# 1점대로 덧셈을 해보자

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3)
# Tensor("Add:0", shape=(), dtype=float32)
# 우리가 원하는 값이 안 나왔다. 세션 통과하게 하자

sess = tf.Session()
print('sess.run([node1, node2]): ', sess.run([node1, node2]))
print('sess.run(node3): ', sess.run(node3))
# sess.run([node1, node2]):  [3.0, 4.0]
# sess.run(node3):  7.0

## 왜 귀찮게 세션을 통과하게 했을까 ?
# tensor: 몇 차원 배열이냐는 뜻. 라틴어 당기다에서 파생했다.
# https://blog.naver.com/complusblog/221237818389
# 다차원 자료인 텐서를 다차원 연산으로 흐르게~ flow 하려고 노드로 만들어서 그래프를 만들어서 연산하게 하려고!
# session을 만들어 sess.run을 하는 것은 프린트만 하려는 것이 아니라 fit과 비슷하다고 생각하면 된다.
## 데이터와 연산식으로 노드를 만들어서 > 그래프를 그려서 > 다차원 연산을 한다
