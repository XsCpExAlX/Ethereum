#!/usr/bin/python3
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
import ccxt
from pandas.core.missing import backfill_1d
import _thread

# Import the RNN packages
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import FALSE


class model_RNN:
    def __init__(self, order_book_range, order_book_window, future_price_window, future_ma_window, num_epochs):
        self.order_book_range = order_book_range
        self.order_book_window = order_book_window
        self.future_price_window = future_price_window
        self.future_ma_window = future_ma_window
        self.num_epochs = num_epochs

  
    def train_and_predict(self, restore=False, data_rnn=None, data_rnn_ckpt=None):
        tf.reset_default_graph()
        print('Restore model? %s' % restore)

        data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/exch_gdax_ethusd_snapshot_20170913.csv')
        data_rnn = data_rnn.drop(new_data_rnn.index[0:2])
        data_rnn = data_rnn[data_rnn.columns.difference(['order_date_time'])]  # .difference() method removes any columns and automatically reorders columns alphanumerically


        # RNN Hyperparams
        # num_epochs is already defined as part of the class
        batch_size = 1
        total_series_length = len(data_rnn.index)
        state_size = 10  # The number of neurons
        num_features = 2 + self.future_price_window + self.order_book_window * 6  # The number of columns to be used for xTrain analysis in RNN
        truncated_backprop_length = 10  # The size of the sequence
        num_classes = 1  # The number of targets to be predicted
        num_batches = int(total_series_length / batch_size / truncated_backprop_length)
        min_test_size = 1000


        # Normalize data
        PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
        PriceMean = data_rnn['trade_px'].mean()
        data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())


        # Pick all apporpriate columns to train and test in RNN
        rnn_column_list = self.get_rnn_column_list()


        # RNN Placeholders

        batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_features],
                                            name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_classes],
                                            name='target_ph')

        # RNN Train-Test Split
        for i in range(min_test_size, len(data_rnn.index)):
            if (i % truncated_backprop_length * batch_size == 0):
                test_first_idx = len(data_rnn.index) - i
                break
        # Purposefully uses self.data_rnn['row_nums'] because self.data_rnn['row_nums'] also becomes normalized
        data_rnn_train = data_rnn_norm[data_rnn['row_num'] < test_first_idx]
        data_rnn_test = data_rnn_norm[data_rnn['row_num'] >= test_first_idx]

        xTrain = data_rnn_train[rnn_column_list].as_matrix()
        yTrain = data_rnn_train[['future_ma_%s' % self.future_ma_window]].as_matrix()
        xTest = data_rnn_test[rnn_column_list].as_matrix()
        yTest = data_rnn_test[['future_ma_%s' % self.future_ma_window]].as_matrix()

        if restore:
            # Weights and Biases In

            weight = tf.get_variable(name='weight', shape=[state_size, num_classes])
            bias = tf.get_variable(name='bias', shape=[num_classes])
            labels_series = tf.unstack(batchY_placeholder, axis=1)  # Unpacking

        else:
            # Weights and Biases In
            weight = tf.Variable(tf.truncated_normal([state_size, num_classes]), name='weight')
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias')
            labels_series = tf.unstack(batchY_placeholder, axis=1)  # Unpacking

        # Forward Pass: Unrolling the cell (input to hidden recurrent layer)
        cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size) # this takes forever!
        states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, dtype=tf.float32)
        states_series = tf.transpose(states_series, [1, 0, 2])

        # Backward Pass: Output
        last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)
        last_label = tf.gather(params=labels_series, indices=len(labels_series) - 1)

        # Prediction, Loss, and Optimizer
        prediction = tf.matmul(last_state, weight) + bias
        loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        loss_list = []
        test_pred_list = []


        # Add saver variable to save and restore all variables from trained model
        saver = tf.train.Saver()


        # Initialize and run session
        with tf.Session() as sess:
            if restore:
                saver.restore(sess, data_rnn_ckpt)

            else:
                tf.global_variables_initializer().run()

                for epoch_idx in range(self.num_epochs):
                    print('Epoch %d' % int(epoch_idx + 1))
                    try:
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * truncated_backprop_length
                            end_idx = start_idx + truncated_backprop_length * batch_size

                            batchX = xTrain[start_idx:end_idx, :].reshape(batch_size, truncated_backprop_length,
                                                                          num_features)
                            batchY = yTrain[start_idx:end_idx].reshape(batch_size, truncated_backprop_length, 1)

                            feed = {batchX_placeholder: batchX, batchY_placeholder: batchY}

                            # TRAIN!
                            _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                                fetches=[loss, train_step, prediction, last_label, prediction],
                                feed_dict=feed)
                            loss_list.append(_loss)

                            if (batch_idx % 1000 == 0):
                                print('Step %d - Loss: %.10f' % (batch_idx, _loss))

                    except ValueError:
                        print('You have reached the end of Training Epoch %d' % int(epoch_idx + 1))
                        pass

                    # Before going on into testing, save variables from trained model to disk
                save_path = saver.save(sess, data_rnn_ckpt)
                print("Model saved in file: %s" % save_path)

                # TEST
                for test_idx in range(len(xTest) - truncated_backprop_length):
                    testBatchX = xTest[test_idx:test_idx + truncated_backprop_length, :].reshape(
                        (1, truncated_backprop_length, num_features))
                    testBatchY = yTest[test_idx:test_idx + truncated_backprop_length].reshape(
                        (1, truncated_backprop_length, 1))

                    # _current_state = np.zeros((batch_size,state_size))
                    feed = {batchX_placeholder: testBatchX,
                            batchY_placeholder: testBatchY}

                    # Test_pred contains 'window_size' predictions, we want the last one
                    _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
                    test_pred_list.append(test_pred[-1][0])  # The last one

            test_pred_list[:] = [(x * PriceRange) + PriceMean for x in test_pred_list]
            yTest[:] = [(x * PriceRange) + PriceMean for x in yTest]

            # Print out real price vs prediction price
            len(test_pred_list)
            predict = pd.DataFrame(test_pred_list, columns=['Prediction'])
            real = pd.DataFrame(yTest, columns=['Price'])
            real_vs_predict = predict.join(real)
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(real_vs_predict)

            # # Print out graphs for loss and Price vs Prediction
            # plt.title('Loss')
            # plt.scatter(x=np.arange(0, len(loss_list)), y=loss_list)
            # plt.xlabel('epochs')
            # plt.ylabel('loss')

            plt.figure(figsize=(15, 6))
            plt.plot(yTest, label='Price', color='blue')
            plt.plot(test_pred_list, label='Predicted', color='red')
            plt.title('Price vs Predicted')
            plt.legend(loc='upper left')
            plt.show()
    

    def get_rnn_column_list(self):
        # Pick all apporpriate columns to train and test in RNN
        rnn_column_list = ['trade_px', 'spread']
        for i in range(1, self.future_price_window + 1):
            rnn_column_list.append('future_price_%s' % i)
        for i in range(1, self.order_book_window + 1):
            rnn_column_list.append('a%s' % i)
            rnn_column_list.append('aq%s' % i)
            rnn_column_list.append('aqq%s' % i)
            rnn_column_list.append('b%s' % i)
            rnn_column_list.append('bq%s' % i)
            rnn_column_list.append('bqq%s' % i)
        return rnn_column_list


if __name__ == '__main__': #TODO: modularize train_and_predict (take out load and rnnCELL), do loop datafetch & predictions. Fetch data by listening to websocket.
    # Read DataFrame.csv

    data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/exch_gdax_ethusd_snapshot_20170913.csv', nrows=2000)
    new_data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/exch_gdax_ethusd_snapshot_20170913.csv', nrows=5000)
    data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/testing1'
    x = model_RNN(30, 1, 5, 5, num_epochs=10)
    x.train_and_predict(restore=False, data_rnn=data_rnn, data_rnn_ckpt=data_rnn_ckpt)
    # x.train_and_predict(restore=True, data_rnn=new_data_rnn, data_rnn_ckpt=data_rnn_ckpt)