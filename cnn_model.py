from __future__ import print_function
import tensorflow as tf
import math
import numpy as np

# NOTE: load the data from the npz
dataset = np.load('./datasets/v0_eigens.npz')

# NOTE: calculate the size of training set and validation set
#       all pre-processed features are inside 'train_eigens'
train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = train_data_size // 5
train_data_size = train_data_size - valid_data_size

# NOTE: split dataset
train_data = dataset['train_eigens'][:train_data_size]
valid_data = dataset['train_eigens'][train_data_size:]

# NOTE: a 896d feature vector for each user, the 28d vector in the end are
#       labels
#       896 = 32 (weeks) x 7 (days a week) x 4 (segments a day) 896+28
train_eigens = train_data[:, :-28]
train_labels = train_data[:, -28:]

valid_eigens = valid_data[:, :-28]
valid_labels = valid_data[:, -28:]

x_train = np.array(train_eigens, dtype="float32")
y_train = np.array(train_labels, dtype="float32")
x_valid = np.array(valid_eigens, dtype="float32")
y_valid = np.array(valid_labels, dtype="float32")

# NOTE: read features of test set
test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 896)
# test_eigens = test_eigens[:, :]

# NOTE: check the shape of the prepared dataset
print('train_eigens.shape = {} {}'.format(x_train.shape, x_train.dtype))
print('train_labels.shape = {}'.format(y_train.shape))
print('valid_eigens.shape = {}'.format(x_valid.shape))
print('valid_labels.shape = {}'.format(y_valid.shape))
print('test_eigens.shape = {}'.format(test_eigens.shape))

train_images = x_train
train_labels = y_train

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 32*28], name = 'xs') # 48*48
    ys = tf.placeholder(tf.float32, [None, 28], name = 'ys')
    lr = tf.placeholder(tf.float32) #For learning rate
    # test flag for batch normalization
    tst = tf.placeholder(tf.bool) 
    iter = tf.placeholder(tf.int32)
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

global_step = tf.Variable(0, name = 'global_step', trainable = False)

def model(xs):
    x = tf.reshape(xs, [-1, 32, 28, 1])
    # conv1 layer
    with tf.name_scope('layer_1'):
        with tf.name_scope('weights'):
            w_conv1_1 = weight_variable([3, 3, 1, 32], 'w_conv1')
        tf.summary.histogram('/weights', w_conv1_1)
        with tf.name_scope('bias'):
            b_conv1_1 = bias_variable([32], 'b_conv1')
        tf.summary.histogram('/bias', b_conv1_1)
        with tf.name_scope('outputs'):
            c_conv1_1 = conv2d(x, w_conv1_1) + b_conv1_1
            h_conv1_1 = tf.nn.leaky_relu(c_conv1_1)
            h_pool1 = max_pool_2x2(h_conv1_1)
        tf.summary.histogram('/outputs', h_pool1)

    # conv2 layer
    with tf.name_scope('layer_2'):
        with tf.name_scope('weights'):
            w_conv2_1 = weight_variable([3, 3, 32, 64], 'w_conv2')
        tf.summary.histogram('/weights', w_conv2_1)
        with tf.name_scope('bias'):
            b_conv2_1 = bias_variable([64], 'b_conv2')
        tf.summary.histogram('/bias', b_conv2_1)
        with tf.name_scope('outputs'):
            c_conv2_1 = conv2d(h_pool1, w_conv2_1) + b_conv2_1
            h_conv2_1 = tf.nn.leaky_relu(c_conv2_1)
            h_pool2 = max_pool_2x2(h_conv2_1)
        tf.summary.histogram('/outputs', h_pool2)

    # conv3 layer
    with tf.name_scope('layer_3'):
        with tf.name_scope('weights_1'):
            w_conv3_1 = weight_variable([3, 3, 64, 128], 'w_conv3_1')
        tf.summary.histogram('/weights_1', w_conv3_1)
        with tf.name_scope('bias_1'):
            b_conv3_1 = bias_variable([128], 'b_conv3_1')
        tf.summary.histogram('/bias_1', b_conv3_1)
        with tf.name_scope('outputs_1'):
            c_conv3_1 = conv2d(h_pool2, w_conv3_1) + b_conv3_1
            h_conv3_1 = tf.nn.leaky_relu(c_conv3_1)
            # h_pool3 = max_pool_2x2(h_conv3_1)
        tf.summary.histogram('/outputs_1', h_conv3_1)

    #func1 layer
    with tf.name_scope('layer_func1'):
        with tf.name_scope('weights'):
            w_f1 = weight_variable([8*7*128, 512], 'w_f1') 
        tf.summary.histogram('/weights', w_f1)
        with tf.name_scope('bias'):
            b_f1 = bias_variable([512], 'b_f1')
        tf.summary.histogram('/bias', b_f1)
        with tf.name_scope('outputs'):
            h_pool4_flat = tf.reshape(h_conv3_1, [-1, 8*7*128])
            h_m1 = tf.matmul(h_pool4_flat, w_f1) + b_f1
            h_f1 = tf.nn.leaky_relu(h_m1)
            h_f1_drop = tf.nn.dropout(h_f1, keep_prob)
        tf.summary.histogram('/outputs', h_f1_drop)

    #func2 layer
    with tf.name_scope('layer_func2'):
        with tf.name_scope('weights'):
            w_f2 = weight_variable([512, 28], 'w_f2')
        tf.summary.histogram('/weights', w_f2)
        with tf.name_scope('bias'):
            b_f2 = bias_variable([28], 'b_f2')
        tf.summary.histogram('/bias', b_f2)
        with tf.name_scope('outputs'):
            h_m2 = tf.matmul(h_f1_drop, w_f2) + b_f2
            prediction = tf.nn.sigmoid(h_m2, name = 'prediction')
        tf.summary.histogram('/outputs', prediction)

    return prediction

prediction = model(xs)
tf.add_to_collection('outputs', prediction)
pre = tf.identity(prediction)
# loss
with tf.name_scope('cross_entropy'):
    pre = tf.clip_by_value(pre,1e-8,tf.reduce_max(pre))
    # cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(pre,1e-10,1.0)))
    cross_entropy = -tf.reduce_mean(ys * tf.log(pre) + (1 - ys) * (tf.log(1 - pre)))
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = pre))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step = global_step)

with tf.name_scope('AUC'):
    # prediction = tf.where(tf.less(prediction, 0.05), tf.zeros_like(prediction), tf.ones_like(prediction))
    auc = tf.metrics.auc(labels = ys, predictions = prediction)
    tf.summary.scalar('auc', auc[1])

batch_size = 64
epochs_completed = 0
index_in_epoch = 0
num_examples = x_train.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

with tf.device('/CPU:0'):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs_2/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs_2/test")  
    saver = tf.train.Saver(max_to_keep=5)
    saver_max_a = 0 
    min_loss = 1000
    for i in range(500001):
        # learning rate decay
        learning_rate = 0.0001
        batch_xs, batch_ys = next_batch(batch_size)
        feed_dict = {xs: batch_xs, ys: batch_ys, keep_prob: 0.5, lr: learning_rate, tst: False}
        ab, _, c, summary = sess.run([auc, train_step, cross_entropy, merged], feed_dict = feed_dict)

        if i % 20 == 0:
            print('auc: ', ab[1], ' cross entropy: ', c)

        if i % 200 == 0:
            train_writer.add_summary(summary, i)
            a, l , result = sess.run([auc, cross_entropy, merged], feed_dict = {xs: x_valid, ys: y_valid, tst: False, iter: i, keep_prob: 1})
            test_writer.add_summary(result, i)
            print('----------------------------------------------------------------------------------------')
            print('valid   auc: ', a[1], ' cross entropy: ', l)
            print('----------------------------------------------------------------------------------------')

            if a[1] >= saver_max_a and l <= min_loss:
                saver.save(sess, './model_2/cnn_{}_{}.ckpt'.format(a[1], l), global_step = global_step)
                print('saved\n')
                saver_max_a = a[1]
                min_loss = l

                # if a[1] > 0.82:
                #     p = sess.run([prediction], feed_dict = {xs: test_eigens, tst: False, iter: i, keep_prob: 1})
                #     write_result('model_1_a{}_{}.csv'.format(a[1], global_step), p)

        
# train_writer.close()  
# test_writer.close() 
sess.close()