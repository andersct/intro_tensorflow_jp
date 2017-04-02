import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras import backend as K
sess = tf.Session()
K.set_session(sess)

from keras.layers import *
from keras.metrics import *
from keras.objectives import categorical_crossentropy


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

# 最適器を選びます
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 変数をイニシャライズします
init_op = tf.global_variables_initializer()
sess.run(init_op)

num_iter = 100
batch_size = 50

# 訓練を始めます
with sess.as_default():
    for iter in range(num_iter):
        batch = mnist_data.train.next_batch(batch_size)
        train_step.run(feed_dict={
            img: batch[0],
            labels: batch[1]
        })

        print('train acc = {:.4f}'.format(
            acc_value.eval(feed_dict={
                img: mnist_data.train.images,
                labels: mnist_data.train.labels
            })
        ))

        print('valid acc = {:.4f}'.format(
            acc_value.eval(feed_dict={
                img: mnist_data.test.images,
                labels: mnist_data.test.labels
            })
        ))
