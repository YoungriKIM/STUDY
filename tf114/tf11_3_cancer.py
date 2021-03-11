# 분류로 데이터 불러와서 만들어라
# 혜지야 고마워

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf 

#[실습]
# accuracy_score로 결론 낼것


#input
x_data, y_data = load_breast_cancer(return_X_y=True)
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) #(569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, shuffle = False, random_state = 66)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30,1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate = 1e-6).minimize(cost)    #1.54521e-5
train = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step  in range(10001):
        cost_val, _, hy_val= sess.run([cost, train, hypothesis], feed_dict = {x:x_train, y:y_train})

        if step % 1000 == 0:
            print(step, 'cost : ', cost_val)

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_test, y:y_test})
    print('hypothesis : ', h, '\n predicted : ', p, '\n accuracy : ', a)  
    print("acc: ",accuracy_score(y_test,p))
    # acc:  0.9122807017543859