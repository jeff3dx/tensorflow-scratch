import tensorflow as tf
import numpy as np
from random import randint
import time
import sys

print(sys.argv)

start = int(time.time())

features = [tf.contrib.layers.real_valued_column("x", dimension=2)]

estimator = tf.contrib.learn.DNNRegressor(
    feature_columns=features,
    hidden_units=[20],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

data_x_train = []
data_y_train = []
data_x_eval = []
data_y_eval = []

for _ in range(1000):
    x1 = randint(1, 100)
    x2 = randint(1, 100)
    data_x_train.append([x1, x2])
    data_y_train.append([x1 * x2])

for _ in range(100):
    a1 = randint(1, 100)
    a2 = randint(1, 100)
    data_x_eval.append([a1, a2])
    data_y_eval.append([a1 * a2])

x_train = np.array(data_x_train, dtype=np.int)
y_train = np.array(data_y_train, dtype=np.int)
x_eval = np.array(data_x_eval, dtype=np.int)
y_eval = np.array(data_y_eval, dtype=np.int)

print("Training...")
estimator.fit(input_fn=tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=1), steps=2000)
print("Done training.")

train_loss = estimator.evaluate(input_fn=tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=1))
eval_loss = estimator.evaluate(input_fn=tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size=1))

# get args and make prediction input
user_a = int(sys.argv[1])
user_b = int(sys.argv[2])
x_user = np.array([user_a, user_b], dtype=np.int)

def user_input():
    return { "x": np.array([[user_a, user_b]], dtype=np.int) }

nn_answer = list(estimator.predict_scores(input_fn=user_input))
user_answer = user_a * user_b

elapsed = int(time.time()) - start


print("")
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
print("")
print("user answer: %r"% user_answer)
print("nn   answer: %r"% nn_answer)
print("")
print("seconds: %r"% elapsed)
print("")
