# https://www.tensorflow.org/get_started/mnist/beginners
from random import randint
from tensorflow.examples.tutorials.mnist import input_data
# training_set = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

training = []
answers = []
for _ in range(1000):
    x1 = randint(1, 10)
    x2 = randint(1, 10)
    training.append([x1, x2])
    answers.append([x1 * x2])

# inputs
# placeholder floats, 1000 rows, 2 values each
x = tf.placeholder(tf.float32, [1000, 2])

# layer, 10 neurons, weights and biases
W = tf.Variable(tf.zeros([2, 10]))
b = tf.Variable(tf.zeros([10]))

# implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for the correct answers, 1000 rows, 1 value
y_ = tf.placeholder(tf.float32, [1000, 1])

# cross entropy calculates the error/cost/badness
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# use gradient decent with a rate of 0.5
train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# launch interactive session
sess = tf.InteractiveSession()

# initialize the variables
tf.global_variables_initializer().run()

# run the training step 1000 times
# each loop gets 100 random training points from the training set,
# stochastic gradient descent, cheaper than evaluating ALL trainging data on every loop, but nearly as effective
for _ in range(100):
    sess.run(train, {x: training, y: answers})

# check if prediction equals the truth
# produces array of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# to get accuracy cast booleans to floats ([true, true, false] -> [1.0, 1.0, 0.0])
# and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy)


