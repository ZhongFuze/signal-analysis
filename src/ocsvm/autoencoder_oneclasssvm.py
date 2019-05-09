#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# 参数
learning_rate = 0.008
training_epochs = 130
batch_size = 2560


# 编码器
def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def one_class_learning(dataset, testset, n_input, one_class_label, filename):
    n_hidden_1 = int(n_input / 2)
    n_hidden_2 = int(n_input / 4)

    # 输入设置为placeholder
    X = tf.placeholder("float", [None, n_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }

    # 构建模型
    encoder_op = encoder(X, weights, biases)
    decoder_op = decoder(encoder_op, weights, biases)

    # 模型预测
    y_pred = decoder_op
    y_true = X

    # 损失函数
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

    # 优化
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # 初始化参数
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(len(dataset['data']) / batch_size)
        # 训练
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs = dataset['data'][i * batch_size:(i + 1) * batch_size]
                batch_ys = dataset['label'][i * batch_size:(i + 1) * batch_size]
                # 正常label
                batch_xs = [batch_xs[j] for j in range(len(batch_xs)) if batch_ys[j] == one_class_label]
                o, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        encode_decode = sess.run(y_pred, feed_dict={X: testset['data']})
        # examples_to_show = 14
        # f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
        # for i in range(examples_to_show):
        #    print(testset['label'][i],sess.run(tf.reduce_mean(tf.pow(testset['data'][i] - encode_decode[i], 2))))
        #    a[0][i].imshow(np.reshape(testset['data'][i], (28, 28)))
        #    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # f.show()
        # plt.draw()
        # plt.waitforbuttonpress()

        wf = open(filename, 'a+')
        for i in range(len(encode_decode)):
            wf.write(str(one_class_label) + ',' + str(testset['label'][i]) + ','
                     + str(sess.run(tf.reduce_mean(tf.pow(testset['data'][i] - encode_decode[i], 2)))) + '\n')
            if i % 500 == 0:
                print(i)
        wf.close()


def decode_one_hot(label):
    return max([i for i in range(len(label)) if label[i] == 1])


def mnist_test():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    trainset = {'data': mnist.train.images, 'label': [decode_one_hot(label) for label in mnist.train.labels]}
    testset = {'data': mnist.test.images, 'label': [decode_one_hot(label) for label in mnist.test.labels]}
    one_class_learning(trainset, testset, 784, 7, 'label_7.csv')


if __name__ == '__main__':
    mnist_test()
