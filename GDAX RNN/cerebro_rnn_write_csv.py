from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from datetime import datetime
import urllib3
import string
import os
import sys
import time
import ccxt
#from bitflyer_real_api import access_key, secret_key
#from gdax_real_api import real_api_key, real_secret_key, real_passphrase


# Import the RNN packages
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decimal
from timeit import default_timer as timer
from backtrader import cerebro


class VariableHolder:  # variable holder
    num_classes = None
    truncated_backprop_length = None
    num_features = None
    last_state = None
    last_label = None
    prediction = None
    batchX_placeholder = None
    batchY_placeholder = None

    def __init__(self, numClasses, truncatedBackdropLength, numFeatures, lastState, lastLabel, Prediction,
                 batchXPlaceholder, batchYPlaceholder):
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


    def __init__(self, orderbook_range, orderbook_window, future_price_window, num_epochs):
        self.orderbook_range = orderbook_range
        self.orderbook_window = orderbook_window
        self.future_price_window = future_price_window
        self.num_epochs = num_epochs


    def process_data(self, restore, data_rnn, resample_freq=None, normalization_factor=1.2):
        # Drop first two rows and set index to trades_date_time
        data_rnn = data_rnn.drop(data_rnn.head(2).index)
        data_rnn['trades_date_time'] = pd.to_datetime(data_rnn['trades_date_time'])
        data_rnn = data_rnn[data_rnn.columns.difference(['order_date_time'])]
        data_rnn = data_rnn.set_index('trades_date_time')#, drop=False)

        # Generate cumsum_aq# and cumsum_bq#
        data_rnn['cumsum_aq1'] = data_rnn['aq1']
        data_rnn['cumsum_bq1'] = data_rnn['bq1']


        # # Generate trade_volume_buys and trade_volume_sells based on update_type
        data_rnn['trade_volume_buys'] = data_rnn['update_type']
        data_rnn['trade_volume_sells'] = data_rnn['update_type']
        if data_rnn['update_type'].dtypes == 'float64':
            data_rnn['trade_volume_buys'] = data_rnn['trade_volume_buys'].map({1: 1, 2: 0}).multiply(data_rnn['trade_volume'])
            data_rnn['trade_volume_sells'] = data_rnn['trade_volume_sells'].map({1: 0, 2: 1}).multiply(data_rnn['trade_volume'])
        else:
            data_rnn['trade_volume_buys'] = data_rnn['trade_volume_buys'].map({'buy': 1, 'sell': 0}).multiply(
                data_rnn['trade_volume'])
            data_rnn['trade_volume_sells'] = data_rnn['trade_volume_sells'].map({'buy': 0, 'sell': 1}).multiply(
                data_rnn['trade_volume'])

        # Generate dictionary for resampling (or groupby)
        d1 = {'a%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
        d2 = {'b%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
        d3 = {'trade_px': ['mean'], 'trade_volume': ['sum'], 'trade_volume_buys': ['sum'], 'trade_volume_sells': ['sum']}
        d4 = {'cumsum_aq1': ['mean']}
        d5 = {'cumsum_bq1': ['mean']}
        d6 = {'aq%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
        d7 = {'bq%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
        d1.update(d2)
        d1.update(d3)
        d1.update(d4)
        d1.update(d5)
        d1.update(d6)
        d1.update(d7)


        # data_rnn = data_rnn.resample(resample_freq).agg(d1)
        data_rnn = data_rnn.groupby(pd.Grouper(freq=resample_freq)).agg(d1).interpolate(method='time')
        data_rnn.columns = data_rnn.columns.droplevel(level=1)

        data_rnn['trade_volume_buys_minus_asks'] = data_rnn['trade_volume_buys'] - data_rnn['cumsum_aq1']
        data_rnn['trade_volume_sells_minus_bids'] = data_rnn['trade_volume_sells'] - data_rnn['cumsum_bq1']
        data_rnn['orderbook_market_strength'] = data_rnn['trade_volume_buys_minus_asks'] - data_rnn['trade_volume_sells_minus_bids']

        data_rnn['orderbook_market_strength_pct_change'] = data_rnn['orderbook_market_strength'].pct_change(periods=self.future_price_window)
        data_rnn['trade_px_pct_change'] = data_rnn['trade_px'].pct_change(periods=self.future_price_window)
        data_rnn = data_rnn.drop(data_rnn.head(self.future_price_window).index) # This is to prevent pct_change from producing NaN for loss



        # future_price is purposely calculated after resampling
        if restore:
            data_rnn['future_price_%s' % (self.future_price_window)] = data_rnn['trade_px']
        else:
            data_rnn['future_price_%s' % (self.future_price_window)] = data_rnn['trade_px'][::-1].rolling(window=self.future_price_window).mean()[::-1]


        # if 'row_num' not in data_rnn.columns: #TODO: add rownum for new incoming data
        data_rnn.insert(0, 'row_num', range(len(data_rnn.index)))  # surrogate for the row number

        # Normalize data
        PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
        PriceMean = data_rnn['trade_px'].mean()
        data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())
        data_rnn =data_rnn.reset_index()
        return PriceRange, PriceMean, data_rnn_norm, data_rnn

    def train_and_predict(self, restore=None, live_trading=None, data_rnn=None, data_rnn_ckpt=None, resample_freq=None, update_freq=None, normalization_factor=None, exchange=None, symbol=None,
                          delay=None, data_window=None, maperiod1=None, maperiod2=None, comm=None, percent=None, order_valid=None):
        tf.reset_default_graph()
        print('Restore model? %s' % restore)

        # RNN Hyperparams
        # num_epochs is already defined as part of the class
        batch_size = 1
        total_series_length = len(data_rnn.index)
        truncated_backprop_length = self.future_price_window  # The size of the sequence
        state_size = 10  # The number of neurons
        num_features = 3  # The number of columns to be used for xTrain analysis in RNN
        num_classes = 1  # The number of targets to be predicted
        num_batches = int(total_series_length / batch_size / truncated_backprop_length)
        min_test_size = 100

        PriceRange, PriceMean, data_rnn_norm, data_rnn_processed = self.process_data(restore, data_rnn, resample_freq=resample_freq)

        rnn_column_list = self.get_rnn_column_list(restore)

        # RNN Placeholders
        batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_features],
                                            name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_classes],
                                            name='target_ph')

        # RNN Train-Test Split
        test_first_idx = 0  # needed for train
        if restore:
            data_rnn_test = data_rnn_norm
            print('nrows of testing = %s' % len(data_rnn_test.index))
        else:
            for i in range(min_test_size, len(data_rnn_processed.index)):
                if (i % truncated_backprop_length * batch_size == 0 and i > int(0.1*len(data_rnn_processed.index))):
                    test_first_idx = len(data_rnn_processed.index) - i
                    break
            # Purposefully uses data_rnn['row_nums'] because data_rnn['row_nums'] also becomes normalized
            print('nrows of data set = %s' % len(data_rnn_processed.index))
            data_rnn_train = data_rnn_norm[data_rnn_processed['row_num'] < test_first_idx]
            print('nrows of training = %s' % len(data_rnn_train.index))

            data_rnn_test = data_rnn_norm[data_rnn_processed['row_num'] >= test_first_idx]
            print('nrows of testing = %s' % len(data_rnn_test.index))

            xTrain = data_rnn_train[rnn_column_list].as_matrix()
            yTrain = data_rnn_train[['future_price_%s' % self.future_price_window]].as_matrix()
        xTest = data_rnn_test[rnn_column_list].as_matrix()
        yTest = data_rnn_test[['future_price_%s' % self.future_price_window]].as_matrix()

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
        cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)  # this takes forever!
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

        # # Generate Dataframes for plotting predictions
        # actual_price_df = pd.DataFrame(columns=['actual_price'])
        # test_pred_list_price_df = pd.DataFrame(columns=['test_pred_list_price'])
        # yTest_price_df = pd.DataFrame(columns=['yTest_price'])

        # Add saver variable to save and restore all variables from trained model
        saver = tf.train.Saver()


        # Initialize and run session
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if restore:
                saver.restore(sess, data_rnn_ckpt)
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
                actual_price_df = data_rnn_processed['trade_px'].tail(len(data_rnn_test.index)).reset_index(drop=True)
                test_pred_list_price_df = test_pred_list
                yTest_price_df = yTest

                # Generate csv compatible with Cerebro (trades_date_time, open, high, low, close, RNN, pct_change, orderbook_market_strength)
                column_list = ['datetime', 'open', 'high', 'low', 'close', 'RNN']
#                yTest_price_df = pd.DataFrame(np.array(yTest).reshape(len(yTest), 1), columns=['yTest_price'], index=cerebro_data_rnn.index)
#                cerebro_data_rnn = cerebro_data_rnn.drop(cerebro_data_rnn.tail(truncated_backprop_length).index)
                test_pred_list_price_df = pd.DataFrame(np.array(test_pred_list).reshape(len(test_pred_list), 1), columns=['test_pred_list_price'])
#                test_pred_list_price_df = test_pred_list_price_df.set_index(cerebro_data_rnn.index)
                cerebro_data_rnn = pd.DataFrame(columns=column_list)
                input(data_rnn_processed.columns)
                cerebro_data_rnn['datetime'] = data_rnn_processed['trades_date_time']
                cerebro_data_rnn['open'] = data_rnn_processed['trade_px']
                cerebro_data_rnn['high'] = data_rnn_processed['trade_px']
                cerebro_data_rnn['low'] = data_rnn_processed['trade_px']
                cerebro_data_rnn['close'] = data_rnn_processed['trade_px']
                cerebro_data_rnn['RNN'] = test_pred_list_price_df
                cerebro_data_rnn.to_csv('C:/Users/Joe/Documents/cerebro_testing.csv')
                input("Done")
                # test_pred_list_price_df.to_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/test_pred_list_price_df.csv')

            else:
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

                            if (batch_idx % 100 == 0):
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
                actual_price_df = data_rnn_processed['trade_px'].tail(len(data_rnn_test.index)).reset_index(drop=True)
                test_pred_list_price_df = test_pred_list
                yTest_price_df = yTest

        self.plot_predictions(actual_price_df, yTest_price_df, test_pred_list_price_df)


    def get_rnn_column_list(self, restore):
        # Pick all apporpriate columns to train and test in RNN
        rnn_column_list = ['trade_px', 'trade_px_pct_change', 'orderbook_market_strength']

        return rnn_column_list


    def plot_predictions(self, plot1=None, plot2=None, plot3=None):
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(plot1, label='actual_price', color='blue')
        ax1.plot(plot2, label='yTest_price', color='red')
        ax1.plot(plot3, label='predicted_future_price_%s' % (self.future_price_window), color='green')
        plt.title('%s actual_price vs yTest_price vs predicted_future_price_%s (resample_freq: %s, num_epochs: %s)' % (symbol, self.future_price_window, resample_freq, num_epochs))
        plt.legend(loc='upper left')

        # ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        # ax2.plot(data_rnn['future_trade_volume_buys_%s' % (self.future_price_window)] * 1,
        #          label='future_trade_volume_buys_%s' % (self.future_price_window), color='green')
        # ax2.legend(loc='lower right')

        plt.show()


if __name__ == '__main__':
    # Instantiate real gdax api to get live data feed
#     trade_exch = ccxt.gdax({'apiKey': real_api_key,
#                             'secret': real_secret_key,
#                             'password': real_passphrase,
#                             'nonce': ccxt.gdax.seconds,
#                             'verbose': False})  # If verbose is True, log HTTP requests
    trade_exch = ccxt.gdax()
    trade_exch.urls['api'] = 'https://api.gdax.com'

    # Trading parameters
    delay = 1
    orderbook_range = 5
    orderbook_window = 1
    future_price_window = 10 # Use 20 for optimal training
    num_epochs = 30
    resample_freq = '20s' # Use '20s' when restore=False for optimal training
    update_freq = float(resample_freq[:-1]) / delay # This is used to tell update_data() to add that many new rows of data before deleting the row with oldest data
    normalization_factor = 1.00 # This is a rescaling factor to create a bigger window for training data set
    symbol = 'ETH/USD'
    data_window = int(future_price_window * update_freq) # Use at least 40 to safely avoid len(test_pred_list) = 0
    maperiod1 = 5
    maperiod2 = 20
    comm = 0.0000  # Market trades are 0.25% or 0.0025
    percent = 0.90  # Choose a value between 0 and 1
    order_valid = data_window#60  # Time allowed for a limit trade order to stay opened
    restore = True
    live_trading = False


    print('Loading %s Market...' % symbol)
    trade_exch.load_markets(symbol)  # request markets

    # Create RNN object
    x = model_RNN(orderbook_range, orderbook_window, future_price_window, num_epochs)

    # Read csv file
    # data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/50_exch_gdax_btcusd_snapshot_20180112/exch_gdax_btcusd_snapshot_20180112.csv')
    # data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/50_exch_gdax_ethusd_snapshot_20180112/exch_gdax_ethusd_snapshot_20180112.csv')
    data_rnn = pd.read_csv('C:/Users/Joe/Documents/GitHub/Ethereum/GDAX RNN/exch_gdax_ethusd_snapshot_20180204.csv')
    # data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/50_exch_gdax_ltcusd_snapshot_20180112/exch_gdax_ltcusd_snapshot_20180112.csv')


    # Provide appropriate ckpt file
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/btc_test'
    data_rnn_ckpt = 'C:/Users/Joe/Documents/GitHub/Ethereum/GDAX RNN/rnn_saved_models/eth_cerebro'
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/ltc_test'
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/test'


    tf.reset_default_graph()

    # Process Data
    x.train_and_predict(restore=restore, live_trading=live_trading, data_rnn=data_rnn, data_rnn_ckpt=data_rnn_ckpt, resample_freq=resample_freq,
                        update_freq=update_freq, normalization_factor=normalization_factor, exchange=trade_exch, symbol=symbol, delay=delay)
