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
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from timeit import default_timer as timer

class VariableHolder: #variable holder
     # num_classes, truncated_backprop_length, num_features, last_state, last_label, prediction, batchX_placeholder, batchY_placeholder
    num_classes = None
    truncated_backprop_length = None
    num_features = None
    last_state = None
    last_label = None
    prediction = None
    batchX_placeholder = None
    batchY_placeholder = None
        
    def __init__(self, numClasses, truncatedBackdropLength, numFeatures, lastState, lastLabel, Prediction, batchXPlaceholder, batchYPlaceholder):
        self.num_classes = numClasses
        self.truncated_backprop_length = truncatedBackdropLength
        self.num_features = numFeatures
        self.last_state = lastState
        self.last_label = lastLabel
        self.prediction = Prediction
        self.batchX_placeholder = batchXPlaceholder
        self.batchY_placeholder = batchYPlaceholder

class model_RNN:

    last_trade_id = 0

    def __init__(self, order_book_range, order_book_window, future_price_window, future_ma_window, num_epochs):
        self.order_book_range = order_book_range
        self.order_book_window = order_book_window
        self.future_price_window = future_price_window
        self.future_ma_window = future_ma_window
        self.num_epochs = num_epochs
        
    def process_data(self, data_rnn, restore):
        #print('nrows of raw data= %s' % len(data_rnn.index))

        # Convert 'trades_date_time' and  'order_date_time' from object to datetime object
        data_rnn['trades_date_time'] = pd.to_datetime(data_rnn['trades_date_time'])

        # Add any columns that are needed

        data_rnn['spread'] = data_rnn['a1'] - data_rnn['b1']

        for i in range(1, self.order_book_range + 1):
            data_rnn['aqq%s' % i] = data_rnn[['a%s' % i, 'aq%s' % i]].product(axis=1)
            data_rnn['bqq%s' % i] = data_rnn[['b%s' % i, 'bq%s' % i]].product(axis=1)

        for i in range(1, self.future_price_window + 1):
            data_rnn['future_price_%s' % i] = data_rnn['trade_px']
            data_rnn['future_ma_%s' % self.future_ma_window] = data_rnn['trade_px'][::-1].rolling(window=self.future_price_window).mean()[::-1]

        # Resample data by setting index to 'trades_date_time' to avoid repeat
        data_rnn = data_rnn[data_rnn.columns.difference(['order_date_time'])]  # .difference() method removes any columns and automatically reorders columns alphanumerically
        data_rnn = data_rnn.resample('S', on='trades_date_time').mean().interpolate(method='linear')
        #print('nrows with resample = %s' % len(data_rnn.index))

        #if 'row_num' not in data_rnn.columns: #TODO: add rownum for new incoming data
        data_rnn.insert(0, 'row_num', range(len(data_rnn.index)))  # This column is used as a surrogate for the row number

        # Normalize data
        PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
        PriceMean = data_rnn['trade_px'].mean()
        data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())

        return PriceRange,PriceMean,data_rnn_norm,data_rnn

    def initializeVariables(self, sess, data_rnn):#, cell):
        # RNN Hyperparams
        # num_epochs is already defined as part of the class
        batch_size = 1
        total_series_length = len(data_rnn.index)
        truncated_backprop_length = 5  # The size of the sequence
        state_size = 10  # The number of neurons
        num_features = 2 + self.future_price_window + self.order_book_window * 6  # The number of columns to be used for xTrain analysis in RNN
        num_classes = 1  # The number of targets to be predicted
        num_batches = int(total_series_length / batch_size / truncated_backprop_length)
        min_test_size = 1000

        # RNN Placeholders
        batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_features],
                                            name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_classes],
                                            name='target_ph')

        PriceRange,PriceMean,data_rnn_norm = self.process_data(data_rnn)

        rnn_column_list = self.get_rnn_column_list()

        # RNN Train-Test Split
        data_rnn_test = data_rnn_norm
        #print('nrows of testing = %s' % len(data_rnn_test.index))

        xTest = data_rnn_test[rnn_column_list].as_matrix()
        yTest = data_rnn_test[['future_ma_%s' % self.future_ma_window]].as_matrix()

        # Weights and Biases In
        weight = tf.get_variable(name='weight', shape=[state_size, num_classes])
        bias = tf.get_variable(name='bias', shape=[num_classes])
        labels_series = tf.unstack(batchY_placeholder, axis=1)  # Unpacking

        # Forward Pass: Unrolling the cell (input to hidden recurrent layer)
        cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size) # this takes forever!
        states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, dtype=tf.float32)
        states_series = tf.transpose(states_series, [1, 0, 2])

        # Backward Pass: Output
        last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)
        last_label = tf.gather(params=labels_series, indices=len(labels_series) - 1)

        # Prediction
        prediction = tf.matmul(last_state, weight) + bias

        # Add saver variable to save and restore all variables from trained model
        vh = VariableHolder(num_classes, truncated_backprop_length, num_features, last_state, last_label, prediction, batchX_placeholder, batchY_placeholder)
        saver = tf.train.Saver()
        return saver, data_rnn_norm, vh

    def predict1(self, sess, data_rnn_norm, vh):
        # Initialize and run session
        test_pred_list = self.predict(sess, data_rnn_norm, vh.num_classes, vh.truncated_backprop_length, vh.num_features, vh.last_state, vh.last_label, vh.prediction, vh.batchX_placeholder, vh.batchY_placeholder)
        return test_pred_list

    def test_predict(self, test_pred_list):
        test_pred_list[:] = [(x * PriceRange) + PriceMean for x in test_pred_list]
        yTest[:] = [(x * PriceRange) + PriceMean for x in yTest]

        # Print out real price vs prediction price
        predict = pd.DataFrame(test_pred_list, columns=['Prediction'])
        real = pd.DataFrame(yTest, columns=['Price'])
        real_vs_predict = predict.join(real)
        with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            print(real_vs_predict)

        plt.figure(figsize=(15, 6))
        plt.plot(yTest, label='Price', color='blue')
        plt.plot(test_pred_list, label='Predicted', color='red')
        plt.title('Price vs Predicted')
        plt.legend(loc='upper left')
        plt.show()

    def test(self, data_rnn, data_rnn_ckpt):
        with tf.Session() as sess:
            #cell = tf.contrib.rnn.BasicRNNCell(num_units=1) # this takes forever!
            saver, data_rnn_norm, vh = self.initializeVariables(sess, data_rnn)# cell)
            saver.restore(sess, data_rnn_ckpt)
            self.predict1(sess, data_rnn_norm, vh)
        print("end")
        

    def train_and_predict(self, restore=False, data_rnn=None, data_rnn_ckpt=None, cell=None):
        #tf.reset_default_graph()
        print('Restore model? %s' % restore)

        # RNN Hyperparams
        # num_epochs is already defined as part of the class
        batch_size = 1
        total_series_length = len(data_rnn.index)
        truncated_backprop_length = 5  # The size of the sequence
        state_size = 10  # The number of neurons
        num_features = 2 + self.future_price_window + self.order_book_window * 6  # The number of columns to be used for xTrain analysis in RNN
        num_classes = 1  # The number of targets to be predicted
        num_batches = int(total_series_length / batch_size / truncated_backprop_length)
        min_test_size = 1000

        PriceRange,PriceMean,data_rnn_norm,data_rnn_processed = self.process_data(data_rnn, restore)

        rnn_column_list = self.get_rnn_column_list()

        # RNN Placeholders
        batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_features],
                                            name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_classes],
                                            name='target_ph')

        # RNN Train-Test Split
        test_first_idx = 0 # needed for train
        if restore:
            data_rnn_test = data_rnn_norm
            #print('nrows of testing = %s' % len(data_rnn_test.index))

        else:
            for i in range(min_test_size, len(data_rnn_processed.index)):
                if (i % truncated_backprop_length * batch_size == 0):
                    test_first_idx = len(data_rnn_processed.index) - i
                    break
            # Purposefully uses data_rnn['row_nums'] because data_rnn['row_nums'] also becomes normalized
            print(data_rnn_processed.columns)
            print(test_first_idx)
            print(len(data_rnn_norm))
            data_rnn_train = data_rnn_norm[data_rnn_processed['row_num'] < test_first_idx]
            #data_rnn_train = data_rnn_norm[data_rnn['row_num'] < (test_first_idx-2400)]
            print('nrows of training = %s' % len(data_rnn_train.index))
            data_rnn_test = data_rnn_norm[data_rnn_processed['row_num'] >= test_first_idx]
            print('nrows of testing = %s' % len(data_rnn_test.index))

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
                exchange = ccxt.bitflyer()
                predicted_price = 0
                while True:
                    #input(data_rnn['trades_date_time'])
                    start_time = timer()
                    updated,updated_data_rnn = self.updateData(exchange, data_rnn)
                    if updated:
                        data_rnn = updated_data_rnn
                        #input(data_rnn['trades_date_time']) 
                        test_pred_list = []

                        PriceRange, PriceMean, data_rnn_norm, data_rnn_processed = self.process_data(data_rnn, restore)
                        xTest = data_rnn_norm[self.get_rnn_column_list()].as_matrix()
                        yTest = data_rnn_norm[['future_ma_%s' % self.future_ma_window]].as_matrix()

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

                        actual_price = data_rnn.tail(1)['trade_px']
                        print("actual price - predicted price: ", actual_price.item() - predicted_price) 
                        predicted_price = test_pred_list[-1] * PriceRange + PriceMean

                        print("time taken for iteration: ", timer()-start_time)
                        print("")
                        #self.plot_predictions(test_pred_list, yTest, PriceRange, PriceMean)

                        time.sleep(1)
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
            
            self.plot_predictions(test_pred_list, yTest, PriceRange, PriceMean)


    def plot_predictions(self, test_pred_list, yTest, PriceRange, PriceMean):
        test_pred_list[:] = [(x * PriceRange) + PriceMean for x in test_pred_list]
        yTest[:] = [(x * PriceRange) + PriceMean for x in yTest]

        # Print out real price vs prediction price
        len(test_pred_list)
        predict = pd.DataFrame(test_pred_list, columns=['Prediction'])
        real = pd.DataFrame(yTest, columns=['Price'])
        real_vs_predict = predict.join(real)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        #    print(real_vs_predict)

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
    
    def predict(self, sess, data_rnn, num_classes, truncated_backprop_length, num_features, last_state, last_label, prediction, batchX_placeholder, batchY_placeholder):
        #truncated_backprop_length=5
        #num_features=20
        print('Model restored from file: %s' % data_rnn_ckpt)
        # print('weight: %s' % weight.eval())
        # print('bias: %s' % bias.eval())

        xTest = data_rnn[self.get_rnn_column_list()].as_matrix()
        yTest = data_rnn[['future_ma_%s' % self.future_ma_window]].as_matrix()

        test_pred_list = []
        # Predict new_test_set with previously saved model
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

        return test_pred_list


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

    def updateData(self, exchange, data_rnn):
        new_data_rnn, new_trade_id = self.fetchExchangeData(exchange)
        if new_trade_id != self.last_trade_id:
            #input("update")
            data_rnn = data_rnn.drop(data_rnn.head(1).index)
            data_rnn = pd.concat([data_rnn, new_data_rnn])
            data_rnn = data_rnn.reset_index(drop=True)
            #input(data_rnn)
            return True,data_rnn

        return False,data_rnn

    def preloadData(self, iteration = 100, interval = 1):
        print("Starting preload")
        data = self.makeFetchDF()
        exchange = ccxt.bitflyer()

        trade_id = 0
        i = 0
        j = 0
        while i < iteration:
            if i % 20 == 0 and i != j:
                j = i
                print("Currently at iteration ",i)

            trade_data, new_trade_id = self.fetchExchangeData(exchange)
            if new_trade_id != trade_id:
                trade_id = new_trade_id
                data = pd.concat([data,trade_data])
                i += 1
            #data.tail(1)[trade_id]
            time.sleep(interval)

        self.last_trade_id = trade_id
        return data.sort_values(by='trades_date_time'), trade_id
            
    def fetchExchangeData(self, exchange):
        data = self.makeFetchDF()

        #exchange = ccxt.gdax()
        #print(exchange.fetch_markets())
        trades = exchange.fetch_trades('BTC/USD')
        trade = trades[0]

        orderbook = exchange.fetch_order_book('BTC/USD')
        asks = orderbook['asks']
        bids = orderbook['bids']
        
        values = {'trade_px':trade['price'],'update_type':trade['side'],'trades_date_time':trade['datetime']}
        for i in range(1, self.order_book_range + 1):
            values['a%s' % i] = asks[i][0]
            values['aq%s' % i] = asks[i][1]
            values['b%s' % i] = bids[i][0]
            values['bq%s' % i] = bids[i][1]
        return data.append(values, ignore_index=True), trade['id']

    def makeFetchDF(self):
        column_list = ['trade_px', 'trades_date_time', 'update_type']
        for i in range(1, self.order_book_range + 1):
            column_list.append('a%s' % i)
            column_list.append('aq%s' % i)
            column_list.append('b%s' % i)
            column_list.append('bq%s' % i)

        dataFrame = pd.DataFrame(columns=column_list)
        return dataFrame

if __name__ == '__main__': #TODO: modularize train_and_predict (take out load and rnnCELL), Fetch data by listening to websocket.
    # Read DataFrame.csv
    #new_data_rnn = pd.read_csv('C:/Users/Joe/Documents/exch_gdax_ethusd_snapshot_20170913.csv', nrows=20000)
    ''' some data processing done for the loaded csv file that was in train_and_predict(). I took it out here since it is not needed for livedata feeds
    # Drop first few rows of data because for some reason prices are 0.00
    data_rnn = data_rnn.drop(data_rnn.index[0:2])
    print('nrows with dropped initial zeros= %s' % len(data_rnn.index))

    data_rnn = data_rnn[data_rnn.columns.difference(['order_date_time'])]  # .difference() method removes any columns and automatically reorders columns alphanumerically
    '''

    data_rnn_ckpt = "rnn_saved_models/test.ckpt"
    x = model_RNN(order_book_range=5, order_book_window=1, future_price_window=20, future_ma_window=20, num_epochs=50)
    #vh = VH()
    new_data_rnn, trade_id = x.preloadData(100, 0.5)
    #new_data_rnn.to_csv("preload_data.csv")  # for testing. We can save the data from preload and just reuse that for testing so we don't have to wait every execution.
    #new_data_rnn = pd.read_csv("preload_data.csv")

    tf.reset_default_graph()
    #x.test(new_data_rnn, data_rnn_ckpt)

    x.train_and_predict(restore=True, data_rnn=new_data_rnn, data_rnn_ckpt=data_rnn_ckpt, cell=None)
    '''

    trade_id = 0
    while True:
       tf.reset_default_graph()
       trade_data,new_trade_id = x.
)
       if new_trade_id != trade_id:
           trade_id = new_trade_id
           new_data_rnn = new_data_rnn.drop(0) # take out the leftmost
           new_data_rnn = pd.concat([new_data_rnn,trade_data]) #add it to ends
           x.train_and_predict(restore=True, data_rnn=new_data_rnn, data_rnn_ckpt=data_rnn_ckpt, cell=None)
       time.sleep("hi")
    '''