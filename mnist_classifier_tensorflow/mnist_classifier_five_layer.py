#
# Copyright 2017 Vishal Maini
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
- define architecture of NN
- import training data, train/test split
- initialize hyperparameters (weights, bias, learning_rate, batch_size, etc.)
- forward prop
- calculate loss
- backprop
- weight updates
- repeat until convergence
- visualize performance, tune hyperparameters (test with cross-validation)
- deploy on test data
'''

import tensorflow as tf
import tensorflowvisu
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

'''
TensorFlow uses a deferred execution model:
1) Create computation graph in memory using placeholders.
2) Execute Session --> perform actual computations.
'''

# Hyperparameters
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # input tensor from image pixels
Y_ = tf.placeholder(tf.float32,[None,10]) # correct labels
pkeep = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

H1 = 200 # hidden layer neurons
H2 = 100
H3 = 60
H4 = 30
C = 10 # num classes

# 1-layer version:
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# 1-layer yields 92% accuracy. 2-layer (adding 1 hidden layer) --> 97% accuracy

# weight initialization
W1 = tf.Variable(tf.truncated_normal([28*28,H1],stddev=0.1)) #initializes weights randomly w truncated normal distribution
B1 = tf.Variable(tf.ones([H1])/10)

W2 = tf.Variable(tf.truncated_normal([H1,H2],stddev=0.1))
B2 = tf.Variable(tf.ones([H2])/10)

W3 = tf.Variable(tf.truncated_normal([H2,H3],stddev=0.1))
B3 = tf.Variable(tf.ones([H3])/10)

W4 = tf.Variable(tf.truncated_normal([H3,H4],stddev=0.1))
B4 = tf.Variable(tf.ones([H4])/10)

W4 = tf.Variable(tf.truncated_normal([H3,H4],stddev=0.1))
B4 = tf.Variable(tf.zeros([H4])/10)

W5 = tf.Variable(tf.truncated_normal([H4,C],stddev=0.1))
B5 = tf.Variable(tf.zeros([C])/10)

# model with activation functions + dropout applied

Y1 = tf.nn.relu(tf.matmul(tf.reshape(X, [-1, 784]),W1) + B1)
Y1d = tf.nn.dropout(Y1,pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d,W2) + B2)
Y2d = tf.nn.dropout(Y2,pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d,W3) + B3)
Y3d = tf.nn.dropout(Y3,pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d,W4) + B4)
Y4d = tf.nn.dropout(Y4,pkeep)

Y_logits = tf.matmul(Y4d,W5) + B5 # log odds p/(1-p)
Y = tf.nn.softmax(Y_logits)

# X = [NxD] <-- D = product of pixel dimensions
# W1 = [DxH]
# X.W1 + B1 = [NxH]
# W2 = [HxC]
# X.W2 + B2 = Y = [NxC]

# calculate loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_))*100
# alt: cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
# numerically unstable bc log(0) = NaN

# calculate accuracy
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) #cast as float32 to save memory

# visualize weights, bias, inputs, labels
weights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
biases = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X,Y,Y_)
It = tensorflowvisu.tf_format_mnist_images(X,Y,Y_,lines=25)
visualization = tensorflowvisu.MnistDataVis()

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training function
def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # learning rate decay
    max_lr = 0.005
    min_lr = 0.0001
    decay_speed = 2000
    learning_rate = min_lr + (max_lr - min_lr) * np.exp(-i/decay_speed)

    if update_train_data:
        a,c,im,w,b = sess.run([accuracy, cross_entropy,I,weights,biases],{X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print str(i) + ":accuracy: " + str(a) + ", loss: " + str(c) + ", learning rate: " + str(learning_rate)
        visualization.append_training_curves_data(i,a,c)
        visualization.update_image1(im)
        visualization.append_data_histograms(i,w,b)

    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        print str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c)
        visualization.append_test_curves_data(i, a, c)
        visualization.update_image2(im)

    # backprop
    sess.run(train_step,{X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep:0.75})

# to disable visualization, replace visualization.animate with:
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)
visualization.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

print "max test accuracy: " + str(visualization.get_max_test_accuracy())

# a_train,c_train = sess.run([accuracy, cross_entropy], feed_dict = train_data)
# print a_train,c_train
#
# test_data = {X: mnist.test.images, Y_: mnist.test.labels}
# a_test,c_test = sess.run([accuracy, cross_entropy], feed_dict=test_data)
# print "model accuracy: {:.3f}. cross-entropy: {:.3f}".format(a_test, c_test)
