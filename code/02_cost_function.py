# (1)1-1/R
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1]
y_data = [1]

W = tf.placeholder(tf.float32)

#----- a neuron
w = tf.Variable(tf.random_normal([1]))
hypo = x_data * W

#----- learning
cost = (hypo - y_data) ** 2

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
weights = []
costs = []

for i in range(-30, 50):
    w_val = i * 0.1
    w = w_val
    curr_cost = sess.run(cost, feed_dict={W: w_val})
    weights.append(w_val)
    costs.append(curr_cost)

# Show the cost function
plt.plot(weights, costs)
plt.show()

