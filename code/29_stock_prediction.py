import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(data):
    numerator = data - np.min(data, 0)   # 모든 데이터를 0부터 시작하도록
    denominator = np.max(data, 0) - np.min(data, 0)  #최대값과 최소값 차이 구함
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


tf.set_random_seed(777)  # reproducibility


timesteps = seq_length = 7
data_dim = 5
output_dim = 1

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])  # [None, 7, 5]
Y = tf.placeholder(tf.float32, [None, 1])

# Neural Network Definition -------------------
cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True)  # 1차원 출력
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)  # 5차원 데이터가 7번 입력
hypo = outputs[:, -1]  # 7번 입력하여 나온 7번 출력 중 가장 마지막 것을 신경망의 출력으로 사용
# 7묶음 짜리 데이터가 모두 725개, 이만큼 hypo도 만들어짐.

loss = tf.reduce_sum(tf.square(hypo - Y))  # cost/loss
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = normalize(xy)
x = xy
y = xy[:, [-1]]  # Extract label

dataX = []
dataY = []
number_of_label = len(y)

# cvs 파일 전체 데이터를 7줄로 구성된 725개의 묶음으로 나누고 724개의 정답
for i in range(0, number_of_label - seq_length): # 725(732 - 7)
    _x = x[i:i + seq_length] # 0~6  (0~725)
    _y = y[i + seq_length]  # The next close price (6~732)
    print(_x, "->", _y) # _x의 한 묶음(7개)가 입력되어 unfolding되면 _y가 되어야 함.
    dataX.append(_x) # 725(732-7)개의 묶음들이  만들어져 들어감
    dataY.append(_y) # 7번 인덱스에 있는 y값부터 하나씩 꺼내어 구성함. 0~6번 인덱스꺼는 무시

# Preparation of data: 70% for train, 30% for testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX = np.array(dataX[0:train_size])  # from 0 to train_size - 1
trainY = np.array(dataY[0:train_size])

testX = np.array(dataX[train_size:len(dataX)])  # from train_size to len(dataX) - 1
testY = np.array(dataY[train_size:len(dataY)])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train, feed_dict={X: trainX, Y: trainY})
    step_loss = sess.run(loss, feed_dict={X: trainX, Y: trainY})

    print(i, step_loss)

testPredict = sess.run(hypo, feed_dict={X: testX})
print("RMSE", sess.run(rmse, feed_dict={targets: testY, predictions: testPredict}))
plt.plot(testY)
plt.plot(testPredict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()




