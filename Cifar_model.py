import tensorflow as tf
import numpy as np
import Cifar_data_loader
STEPS=10000
BATCH_SIZE=50

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

cifar = Cifar_data_loader.CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv2 = conv_layer(conv1, shape=[5, 5, 32, 64])
conv3 = conv_layer(conv2, shape=[5, 5, 64, 64])
pool_1 = max_pool_2x2(conv3)
batch_norm_1=n1=tf.nn.relu(tf.contrib.layers.batch_norm(pool_1))

conv4 = conv_layer(batch_norm_1, shape=[5, 5, 64, 128])
conv5 = conv_layer(conv4, shape=[5, 5, 128, 256])
conv6 = conv_layer(conv5, shape=[5, 5, 256, 256])
conv_pool_2 = max_pool_2x2(conv6)
batch_norm_2=tf.nn.relu(tf.contrib.layers.batch_norm(conv_pool_2))
conv_flat = tf.reshape(batch_norm_2, [-1, 8 * 8 * 256])

full_1 = tf.nn.relu(full_layer(conv_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_conv,labels= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], 
                                                 keep_prob: 1.0})
                   for i in range(10)])
    print("Accuracy: {:.4}%".format(acc * 100))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = cifar.train.next_batch(BATCH_SIZE)
        print(batch[0].shape)
        if i%200==0:
            train_result=sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], 
                                        keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_result*100))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], 
                                        keep_prob: 0.5})

    test(sess)