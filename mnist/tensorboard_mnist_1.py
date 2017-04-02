import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *
from keras.metrics import *
from keras.objectives import categorical_crossentropy


PATH = os.path.dirname(os.path.abspath(__file__))

# ダウンロードしたデータをMNIST_dataに名付けたディレクトリに保存します
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# MNISTのイメージは28**2　= 784ピクセルで十個カテゴリー
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

# モデルを作ります
x = Dense(32, activation='relu')(img)
preds = Dense(10, activation='softmax')(x)

# 損失を定義します
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# 精度を定義します
acc_value = tf.reduce_mean(categorical_accuracy(labels, preds))

# 記憶された変数を設定します
loss_summary = tf.summary.scalar('loss', loss)
acc_value_summary = tf.summary.scalar('acc', acc_value)
summary_op = tf.summary.merge_all()

# 最適器を選びます
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 変数と定数をイニシャライズします
log_path = os.path.join(PATH, 'logs')
num_iter = 100
batch_size = 50
init_op = tf.global_variables_initializer()
sess.run(init_op)
train_writer = tf.summary.FileWriter(log_path + '/train', graph=sess.graph)
valid_writer = tf.summary.FileWriter(log_path + '/valid', graph=sess.graph)
test_writer = tf.summary.FileWriter(log_path + '/test', graph=sess.graph)

samples_per_epoch = mnist_data.train.labels.shape[0]  # 訓練サンプルの合計
epochs = 10
assert samples_per_epoch % batch_size == 0, \
    'batch size {} does not divide epoch size {}'.format(batch_size, samples_per_epoch)

# 訓練を始めます
with sess.as_default():
    epoch = 0
    while epoch < epochs:
        print('Starting epoch {}'.format(epoch))
        sampled = 0
        while sampled < samples_per_epoch:
            # 訓練
            batch = mnist_data.train.next_batch(batch_size)
            summary, _ = sess.run([summary_op, train_step], feed_dict={
                img: batch[0],
                labels: batch[1]
            })
            train_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

            # 検証
            valid_batch = mnist_data.validation.next_batch(batch_size)
            summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
                img: valid_batch[0],
                labels: valid_batch[1]
            })
            valid_writer.add_summary(summary, epoch*samples_per_epoch + sampled)

            sampled += batch_size
        epoch += 1

        # テスト
        summary, _, _ = sess.run([summary_op, acc_value, loss], feed_dict={
            img: mnist_data.test.images,
            labels: mnist_data.test.labels
        })
        test_writer.add_summary(summary, epoch*samples_per_epoch)


# tensorboard --logdir=`pwd`/mnist/logs
# localhost:6006
