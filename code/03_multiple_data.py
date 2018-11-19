# (3)1-1/R
import tensorflow as tf

from myplot import MyPlot

x_data = [1, 2, 3]
y_data = [1, 2, 3]

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypo = w * x_data + b
#-----

cost = tf.reduce_mean((hypo - y_data) ** 2)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

costs = []

for i in range(1001):
    sess.run(train)

    if i % 50 == 0:
        print(sess.run(w), sess.run(b), sess.run(cost))
        costs.append(sess.run(cost))

print(hypo) #shape=(3,)
print(sess.run(hypo))

p = MyPlot()
p.show_list(costs)

