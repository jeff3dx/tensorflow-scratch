# https://www.tensorflow.org/get_started/mnist/beginners

from tensorflow.examples.tutorials.mnist import input_data

training_set = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# inputs
# placeholder of floats, 2 dimensional, None=any length, 784=length of image data dimension
x = tf.placeholder(tf.float32, [None, 784])

# weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for the correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy calculates the error/cost/badness
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# use gradient decent with a rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# launch interactive session
sess = tf.InteractiveSession()

# initialize the variables
tf.global_variables_initializer().run()

# run the training step 1000 times
# each loop gets 100 random training points from the training set,
# stochastic gradient decent, cheaper than evaluating ALL trainging data on every loop, but nearly as effective
for _ in range(1000):
    batch_xs, batch_ys = training_set.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(train_step)


# check if prediction equals the truth
# produces array of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# to get accuracy cast booleans to floats ([true, true, false] -> [1.0, 1.0, 0.0])
# and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: training_set.test.images, y_: training_set.test.labels}))

