# (4)2b-2-1/C(2)
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[0., 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

#------- 2 neurons + 1 neuron
W1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.random_normal([2]))
output1 = tf.sigmoid(tf.matmul(x_data, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
hypo = tf.sigmoid(tf.matmul(output1, W2) + b2)

#----- learning
cost = -tf.reduce_mean(y_data * tf.log(hypo) + tf.subtract(1., y_data) * tf.log(tf.subtract(1., hypo)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

errors = []
for step in range(20001):
    sess.run(train) #계산그래프, 데이터 4개에 대하여 forward prop. 오류값을 구하고 충분히 작지 않으면

    if step % 500 == 0:
        #print(step, sess.run(cost))
        errors.append(sess.run(cost))

#----- testing(classification)
predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

h = sess.run(hypo)
print("\nHypo: ", h)

p = sess.run(predicted)
print("Predicted: ", p)

a = sess.run(accuracy)
print("Accuracy(%): ", a * 100)

from myplot import MyPlot
p = MyPlot()
p.show_list(errors)

