# (4)2b-4S/C(4)
import tensorflow as tf

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

X = tf.placeholder(tf.float32, [4, 2])
Y = tf.placeholder(tf.float32, [4, 4])

#------- 2 inputs 4 neurons
W = tf.Variable(tf.random_normal([2, 4]))
b = tf.Variable(tf.random_normal([4]))
output = tf.matmul(X, W) + b

#----- learning
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                              labels=Y))

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})

    if step % 500 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

#----- testing(classification)
predicted = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

h = sess.run(output, feed_dict={X:x_data})
print("\nLogits: ", h)

p = sess.run(predicted, feed_dict={X:x_data, Y:y_data})
print("Predicted: ", p)

a = sess.run(accuracy, feed_dict={X:x_data, Y:y_data})
print("Accuracy(%): ", a * 100)
