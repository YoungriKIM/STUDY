# w, cost 의 history를 뽑아 그래프도 그려보자

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [3., 4., 6.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis-y))
# loss = mse 와 같은 의미

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i #* 0.1
        curr_cost = sess.run(cost, feed_dict={w: curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print('==========================================')
print(w_history)
print('==========================================')
print(cost_history)
print('==========================================')

plt.plot(w_history, cost_history)
plt.show()