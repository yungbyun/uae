# (1)1-1/R
import tensorflow as tf

x_data = [1]
y_data = [1]

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = w * x_data

#----- learning
cost = (hypo - y_data) ** 2
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('w:', sess.run(w), 'cost:', sess.run(cost))

for i in range(1001):
    sess.run(train)

    if i % 100 == 0:
        print('w:', sess.run(w), 'cost:', sess.run(cost))

#----- testing(prediction)
x_data = [3]
print(sess.run(hypo))







