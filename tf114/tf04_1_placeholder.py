# 플레이스 홀더에 대해 알아보자

import tensorflow as tf
sess = tf.Session()

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(3.0, tf.float32)
node3 = tf.add(node1, node2)

# ---------------------------------
# 플레이스 홀더를 사용해보자
# 값을 미리 준 것이 아니라, 그래프를 만든 후에 값을 넣어준다고 생각하면 된다. placeholder는 인풋레이어와 비슷하다

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))               # 7.5
# 두개 이상의 값도 가능할까 ?
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))        # [4. 7.]
print(sess.run(adder_node, feed_dict={a:[1,3], b:2}))            # [3. 5.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))            # 18.0