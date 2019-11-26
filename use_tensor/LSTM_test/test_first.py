'''
Created on Sep 23, 2019

copy from https://github.com/huqinwei/tensorflow_demo/blob/master/lstm_mnist/multi_lstm.py

'''

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import os

#hyparam
LR = 0.001#1e-3
STEP_SIZE = 28
INPUT_SIZE = 28
BATCH_SIZE = 100
HIDDEN_CELL = 256
LSTM_LAYER = 3
CLASS_NUM = 10
file_dir = "my_model_saved"
file_name = "/save_net.ckpt"

#mnist load
MNIST = input_data.read_data_sets('./MNIST_data', one_hot = True)
print(MNIST.train.images.shape)
print(MNIST.train.labels.shape)
print(MNIST.test.labels.shape)

#def  network
tf_x = tf.placeholder(dtype = tf.float32, shape = [None,784],name = 'input')
tf_x_reshaped = tf.reshape(tf_x,[-1,STEP_SIZE,INPUT_SIZE])
tf_y = tf.placeholder(dtype = tf.float32, shape = [None,CLASS_NUM])
keep_prob = tf.placeholder(dtype = tf.float32)
batch_size = tf.placeholder(dtype = tf.int32, shape=[])
print(tf_x)

def get_a_cell(i):
    lstm_cell =rnn.BasicLSTMCell(num_units=HIDDEN_CELL, forget_bias = 1.0, state_is_tuple = True, name = 'layer_%s'%i)
    print(type(lstm_cell))
    dropout_wrapped = rnn.DropoutWrapper(cell = lstm_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    return dropout_wrapped

multi_lstm = rnn.MultiRNNCell(cells = [get_a_cell(i) for i in range(LSTM_LAYER)],
                              state_is_tuple=True)#tf.nn.rnn_cell.MultiRNNCell

init_state = multi_lstm.zero_state(batch_size = batch_size, dtype = tf.float32)

#example 1
# outputs, state = tf.nn.dynamic_rnn(multi_lstm, inputs = tf_x_reshaped, initial_state = init_state, time_major = False)
# final_out = outputs[:,-1,:]
# h_state = state[-1][1]

#
# #example 2
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(STEP_SIZE):
        # (cell_output, state) = multi_lstm(tf_x_reshaped[:,timestep,:],state)
        (cell_output, state) = multi_lstm.call(tf_x_reshaped[:,timestep,:],state)
        outputs.append(cell_output)
        # print('cell_output:', cell_output)
h_state = outputs[-1]

#prediction and loss
W = tf.Variable(initial_value = tf.truncated_normal([HIDDEN_CELL, CLASS_NUM], stddev = 0.1 ), dtype = tf.float32)
print(W)
b = tf.Variable(initial_value = tf.constant(0.1, shape = [CLASS_NUM]), dtype = tf.float32)
predictions = tf.nn.softmax(tf.matmul(h_state, W) + b)
#sum   -ylogy^
cross_entropy = -tf.reduce_sum(tf_y * tf.log(predictions))
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy)


right_predictions_num = tf.equal(tf.argmax(predictions, axis = 1), tf.argmax(tf_y, axis = 1))
accuracy = tf.reduce_mean(tf.cast(right_predictions_num, dtype = tf.float32))#tf.cast


#train
# keep_prob
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('graph', graph=sess.graph)
    _batch_size = 5
    saver = tf.train.Saver()
    if  os.path.exists(file_dir) :#restore
        print('exist!!!!!!!!!!!!!!!!!!!!!!!!')
        saver.restore(sess,file_dir + file_name)

    else:#train and save
        print('not exist!!!!!!!!!!!!!!!!!!!!!!!!')
        for i in range(2000):
            x,y = MNIST.train.next_batch(_batch_size)
            _, loss_,outputs_, state_, right_predictions_num_  = \
                sess.run([train_op, cross_entropy,outputs, state,right_predictions_num], {tf_x:x, tf_y:y, keep_prob:1.0,batch_size:_batch_size})
            # print('step ', i, ' loss:', loss_)
            # print('step [{0}] loss:{1}',{i,loss_})
            print('step [{0}] loss:{1}'.format(i,loss_))

            # print('right_predictions_num_:', right_predictions_num_)

            if i % 100 == 0:
                # tensorflow.python.framework.errors_impl.InvalidArgumentError: ConcatOp: Dimensions of inputs should match: shape[0] = [1000, 28] vs.shape[1] = [100, 256]
                # test_x, test_y = MNIST.test.next_batch(BATCH_SIZE * 10)
                total_accuracy = 0.
                total_test_batch = 10
                for j in range(total_test_batch):
                    test_x, test_y = MNIST.test.next_batch(_batch_size)
                    accuracy_ =  sess.run([accuracy], {tf_x:test_x, tf_y:test_y, keep_prob:1.0, batch_size : _batch_size})
                    total_accuracy += accuracy_[0]
                total_accuracy = total_accuracy / total_test_batch
                print('total_accuracy:', total_accuracy)
                # print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))


        X_batch, y_batch = MNIST.test.next_batch(_batch_size)
        print(X_batch.shape, y_batch.shape)
        _outputs, _state = np.array(sess.run([outputs, state],
                                             feed_dict={tf_x:X_batch, tf_y:y_batch,keep_prob:1.0,batch_size:_batch_size}))
        print('_outputs.shape = ', np.asarray(_outputs).shape)
        print('arr_state.shape =', np.asarray(_state).shape)
        # 可见 outputs.shape = [ batch_size, timestep_size, hidden_size]
        # state.shape = [layer_num, 2, batch_size, hidden_size]
        if not os.path.exists(file_dir):  # restore
            print('mkdir!!!!!!!!!!!!!!!!!!!!!!!!!!')
            os.mkdir(file_dir)
        saver.save(sess,file_dir + file_name)
#################################################################################
    print(MNIST.train.labels[10:20])
    X3 = MNIST.train.images[10]
    img3 = X3.reshape([28,28])
    plt.imshow(img3, cmap='gray')
    plt.show()

#####################################################################################
    X3.shape = [-1, 784]
    y_batch = MNIST.train.labels[10]
    y_batch.shape = [-1, CLASS_NUM]

    X3_outputs = np.array(sess.run(outputs, feed_dict={
        tf_x: X3, tf_y: y_batch, keep_prob: 1.0, batch_size: 1}))
    print(X3_outputs.shape)
    X3_outputs.shape = [28, HIDDEN_CELL]
    print(X3_outputs.shape)


    h_W, h_bias = sess.run([W, b], feed_dict={
        tf_x:X3, tf_y: y_batch, keep_prob: 1.0, batch_size: 1})
    h_bias = h_bias.reshape([-1, 10])

    bar_index = range(CLASS_NUM)
    for i in range(X3_outputs.shape[0]):
        plt.subplot(7, 4, i+1)
        X3_h_shate = X3_outputs[i, :].reshape([-1, HIDDEN_CELL])
        pro = sess.run(tf.nn.softmax(tf.matmul(X3_h_shate, h_W) + h_bias))
        plt.bar(bar_index, pro[0], width=0.2 , align='center')
        plt.axis('off')
    plt.show()