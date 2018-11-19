# (1)1-1/R
import tensorflow as tf

x_data = [1]
y_data = [1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * X

#----- learning
cost = (hypo  -Y) ** 2

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})

    if i % 100 == 0:
        print(sess.run(w), sess.run(cost, feed_dict={X:x_data, Y:y_data}))

#----- testing(prediction)
print(sess.run(hypo, feed_dict={X:[3]}))






