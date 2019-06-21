
import tensorflow as tf
import numpy as np
import math
import matplotlib
import os


# input placeholder
_x = tf.placeholder(dtype=tf.float32)
    
# a non-linear activation of the form y = x^2
_y = tf.square(_x)
    
# draw 5 samples from a distribution
in_x = np.random.uniform(0, 100, 5)
    
sess = tf.Session()
with sess.as_default():
    for x in in_x:
        print(sess.run(_y, feed_dict={_x: x}))

# print(in_x)

# 1024 integers randomly sampled from the range 1-5
x_train = np.random.randint(1, 5, 1024)
y_train = np.square(x_train)

# print(x_train)
n_epochs = 20
n_neurons = 128
 
tf.reset_default_graph()

# add for GPU but it also work without
# sess = tf.Session(config=tf.ConfigProto(
#     allow_soft_placement=True, log_device_placement=True))

sess = tf.Session()

with sess.as_default():
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    # Hidden layer with 128 neurons
    w1 = tf.Variable(tf.truncated_normal([1, n_neurons], stddev=0.1))
    b1 = tf.Variable(tf.constant(1.0, shape=[n_neurons]))
	# This is not a standard activation   
    h1 = tf.square(tf.add(tf.multiply(x, w1), b1))
    
	# Output layer
    w2 = tf.Variable(tf.truncated_normal([n_neurons, 1], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0))
 
    prediction = tf.reduce_mean(tf.add(tf.multiply(h1, w2), b2))
    
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
 
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
 
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for idx, x_batch in enumerate(x_train):
            y_batch = y_train[idx]
            _, _step, _pred, _loss = sess.run([train_op, global_step, prediction, loss], feed_dict={x: x_batch, y: y_batch})
            print ("Step: {}, Loss: {}, Value: {}, Prediction: {}".format(_step, _loss, y_batch, int(_pred)))

test_pred = sess.run(prediction, feed_dict={x: 2})
print (test_pred)