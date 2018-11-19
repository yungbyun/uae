# (N) 784-10S/C(10)
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The first dimension of the placeholder is None,
# meaning we can have any number of rows.
# The second dimension is fixed at 784, meaning each row needs to have 784 columns of data.
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# ------- 784 inupts - 10 neurons - Softmax
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
output = tf.matmul(X, W) + b

# ----- learning
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                              labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100

for epoch in range(15):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

# ----- testing(classification)
predicted = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

'test for all the test images(10,000)'
a = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
print("Accuracy(%): ", round(a * 100, 2))

r = random.randint(0, mnist.test.num_examples - 1)
label = sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1))  # extract the ground truth(label)
print("Label: ", label)

answer = sess.run(tf.argmax(output, 1), feed_dict={X: mnist.test.images[r:r + 1]})
print("Predicted: ", answer)

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

