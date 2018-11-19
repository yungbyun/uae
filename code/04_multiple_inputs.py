# (3)2-1/R
import tensorflow as tf

from myplot import MyPlot

x = [[1., 1], [2, 2], [3, 3]]
y = [[1.], [2], [3]]

#----- a neuron
w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
hypo = tf.matmul(x,  w) + b
#-----

cost = tf.reduce_mean((hypo - y) * (hypo - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

costs = []

for i in range(2001):
    sess.run(train)

    if i % 50 == 0:
        print('hypo:', sess.run(hypo), '|', sess.run(w), sess.run(b), sess.run(cost))

        costs.append(sess.run(cost))

hypo2 = tf.matmul([[4.,4]], w) + b
print(sess.run(hypo2))

p = MyPlot()
p.show_list(costs)




