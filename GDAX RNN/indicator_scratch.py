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
import WebsocketClient
import FindSlopes


# Import the RNN packages
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decimal
from timeit import default_timer as timer


class model_RNN:
    last_trade_id = 0
    last_trade_price = 0


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
        data_rnn = data_rnn.set_index('trades_date_time')
        # print(data_rnn.columns)

        # Generate cumsum_aq# and cumsum_bq#
        data_rnn['cumsum_aq1'] = data_rnn['aq1']
        data_rnn['cumsum_bq1'] = data_rnn['bq1']
        for i in range(2, self.orderbook_range + 1):
            data_rnn['cumsum_aq%s' % i] = data_rnn['aq%s' % i] + data_rnn['cumsum_aq%s' % (i - 1)]
            data_rnn['cumsum_bq%s' % i] = data_rnn['bq%s' % i] + data_rnn['cumsum_bq%s' % (i - 1)]

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
        d4 = {'cumsum_aq%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
        d5 = {'cumsum_bq%s' % (k): ['mean'] for k in range(1, self.orderbook_range + 1)}
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

        # data_rnn['orderbook_market_strength_pct_change'] = data_rnn['orderbook_market_strength'].pct_change(periods=self.future_price_window)
        data_rnn['trade_px_pct_change'] = data_rnn['trade_px'].pct_change(periods=self.future_price_window)
        data_rnn = data_rnn.drop(data_rnn.head(self.future_price_window).index) # This is to prevent pct_change from producing NaN for loss



        # future_price is purposely calculated after resampling
        if restore:
            data_rnn = FindSlopes.findPeaks(data_rnn)
            data_rnn['future_price_%s' % (self.future_price_window)] = data_rnn['trade_px']
        else:
            for i in range(1, self.future_price_window + 1):
                data_rnn['future_price_%s' % (i)] = data_rnn['trade_px'][::-1].rolling(window=i).mean()[::-1]


        # if 'row_num' not in data_rnn.columns: #TODO: add rownum for new incoming data
        data_rnn.insert(0, 'row_num', range(len(data_rnn.index)))  # surrogate for the row number

        # Normalize data
        if restore:
            PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
            PriceMean = data_rnn['trade_px'].mean()
            data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())
        else:
            PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
            PriceMean = data_rnn['trade_px'].mean()
            data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())
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
                update_count = 1
                if live_trading:
                    position, account_USD, start_account, account_coin, coin, buy_price, sell_price, data_rnn, coin_symbol, \
                    usd, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase = self.initialize_trade_logic(
                        exchange, data_rnn, symbol, data_window)

                while True:
                    start_time = timer()
                    updated, updated_data_rnn, update_count = self.updateData(exchange, data_rnn, symbol, update_count)

                    if updated:
                        data_rnn = updated_data_rnn
                        test_pred_list = []
                        PriceRange, PriceMean, data_rnn_norm, data_rnn_processed = self.process_data(restore, data_rnn,
                                                                                                     resample_freq=resample_freq)
                        xTest = data_rnn_test[rnn_column_list].as_matrix()
                        yTest = data_rnn_test[['future_price_%s' % self.future_price_window]].as_matrix()

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
                        yTest = pd.DataFrame({'yTest': yTest.tolist()}) # yTest is converted from numpy-array to dataframe
                        print('len(test_pred_list): %s, update_count: %s, resample_freq: %s' % (len(test_pred_list), update_count, resample_freq))
                        # actual_price = data_rnn_processed['trade_px'].last().iloc[0].iloc[0]
                        actual_price = data_rnn_processed['trade_px'].tail(1)
                        yTest_price = yTest['yTest'].iloc[-1][0]
                        test_pred_list_price = test_pred_list[-1]
                        difference = test_pred_list_price - actual_price

                        print('trade_px: %s, yTest_price: %s, test_pred_list_price: %s' % (actual_price), yTest_price, test_pred_list_price)
                        print('Difference: %s' % difference)

                        if live_trading:
                            position, coin, account_USD, buy_price, sell_price, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase = self.limit_trade_logic(
                                exchange, data_rnn, test_pred_list_price, symbol, data_window, maperiod1, maperiod2,
                                delay,
                                position, account_USD,
                                account_coin, coin, buy_price, sell_price, comm, percent, order_valid=order_valid,
                                limit_valid=limit_valid,
                                limit_buy_executed=limit_buy_executed, limit_sell_executed=limit_sell_executed,
                                start_account=start_account, update_count=update_count,
                                percent_increase=percent_increase)

                    time.sleep(delay)
                    print('Time taken for iteration: ', timer() - start_time)
                    print('')
                    # self.plot_predictions(actual_price_df, test_pred_list_price_df, yTest_price_df)

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
        # rnn_column_list = ['trade_px', 'trade_px_pct_change', 'orderbook_market_strength', 'orderbook_market_strength_pct_change']

        # # rnn_column_list purposely excludes future_price_window+1 because that will be the target and thus must avoid double counting
        # for i in range(1, self.future_price_window):
        #     rnn_column_list.append('future_price_%s' % i)

        return rnn_column_list

    # def generate_price_df(self, actual_price=None, actual_price_df=None, test_pred_list_price=None, test_pred_list_price_df=None,
    #                       yTest_price=None, yTest_price_df=None):
    #     actual_price_df.append(actual_price.iloc[0])
    #     test_pred_list_price_df.append(test_pred_list_price.iloc[0])
    #     yTest_price_df.append(yTest_price.iloc[0])
    #
    #     return actual_price_df, test_pred_list_price_df, yTest_price_df

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

    def updateData(self, exchange, data_rnn, symbol, update_count):
        new_data_rnn, new_trade_id = self.fetchExchangeData(exchange, symbol)
        if new_trade_id != self.last_trade_id:
            if update_count % update_freq == 0:
                data_rnn = data_rnn.drop(data_rnn.head(1).index)
                update_count = 0
            data_rnn = pd.concat([data_rnn, new_data_rnn])
            data_rnn = data_rnn.reset_index(drop=True)
            update_count += 1
            return True, data_rnn, update_count

        return False, data_rnn, update_count

    def updateDataWS(self, exchange, data_rnn, symbol):
        price_changed, new_data_rnn = self.fetchNewExchangeData(exchange, symbol)
        # replace oldest if price changed
        if price_changed:
            data_rnn = data_rnn.drop(data_rnn.head(1).index)
            data_rnn = pd.concat([data_rnn, new_data_rnn])
            data_rnn = data_rnn.reset_index(drop=True)
            return True, data_rnn

        # otherwise, update last data if there were any trades happened since last fetch
        if new_data_rnn is not None:
            head = data_rnn.head(1).index
            data_rnn.loc[head] = new_data_rnn
            data_rnn = data_rnn.reset_index(drop=True)

        return False, data_rnn

    def fetchNewExchangeData(self, exchange, symbol):
        # fetch latest trade data from websocket
        values, new_trade_id = self.fetchTradeDataWS()
        if new_trade_id != self.last_trade_id:
            # make the dataframe with orderbook data
            new_data = self.makeFetchDF()
            values.update(self.getOrderBookData(exchange, symbol))
            values['future_price_%s' % (self.future_price_window)] = values['trade_px']
            new_data = new_data.append(values, ignore_index=True)

            # now set last trade id
            self.last_trade_id = new_trade_id

            # if there is a price change, return true
            if self.last_trade_price != values['trade_px']:
                self.last_trade_price = values['trade_px']
                return True, new_data
            else:
                return False, new_data

        return False, None

    def preloadData(self, iteration=None, interval=None, exchange=None, symbol=None):
        print("Starting preload: %s iterations total" % iteration)
        data = self.makeFetchDF()
        start_time = timer()

        trade_id = 0
        i = 1
        j = 1
        while i <= iteration:
            if i % 5 == 0 and i != j:
                j = i
                print("Currently at iteration ", i)
                # print(data)

            trade_data, new_trade_id = self.fetchExchangeData(exchange, symbol)
            if new_trade_id != trade_id:
                trade_id = new_trade_id
                data = pd.concat([data, trade_data])
                i += 1
            # data.tail(1)[trade_id]
            time.sleep(interval)
        data = data.reset_index(drop=True)
        print('Time taken for preload: ', timer() - start_time)

        self.last_trade_id = trade_id
        return data.sort_values(by='trades_date_time'), trade_id

    def fetchExchangeData(self, exchange, symbol):
        data = self.makeFetchDF()

        trades = exchange.fetch_trades(symbol)
        trade = trades[0]
        # print(trade)
        values = {'trade_px': trade['price'], 'update_type': trade['side'], 'trades_date_time': trade['datetime'], 'trade_volume': trade['amount']}
        values['trade_volume'] = trade['amount']
        values['trade_volume_buys'] = trade['amount']
        values['trade_volume_sells'] = trade['amount']

        # values['trade_volume_buys_minus_asks'] = trade['amount']
        # values['trade_volume_sells_minus_bids'] = trade['amount']
        # values['orderbook_market_strength'] = trade['amount']
        #
        # values['orderbook_market_strength_pct_change'] = trade['amount']
        # values['trade_px_pct_change'] = trade['price']

        values.update(self.getOrderBookData(exchange, symbol))

        values['future_price_%s' % (self.future_price_window)] = values['trade_px']

        return data.append(values, ignore_index=True), trade['id']

    def getOrderBookData(self, exchange, symbol):
        orderbook = exchange.fetch_order_book(symbol)
        asks = orderbook['asks']
        bids = orderbook['bids']
        values = {}

        for i in range(1, self.orderbook_range + 1):
            values['a%s' % i] = asks[i][0]
            values['aq%s' % i] = asks[i][1]
            values['b%s' % i] = bids[i][0]
            values['bq%s' % i] = bids[i][1]
            # values['cumsum_aq%s' % i] = values['aq%s' % i]
            # values['cumsum_bq%s' % i] = values['bq%s' % i]

        return values
        #values['future_price_%s' % (self.future_price_window)] = values['trade_px']

    def fetchTradeDataWS(self):
        trade = self.ws.getLastMessage()
        while trade is None:
            trade = self.ws.getLastMessage()
        values = {'trade_px': trade['price'], 'update_type': trade['side'], 'trades_date_time': trade['time'], 'trade_volume': trade['last_size']}
        values['trade_volume'] = trade['last_size']
        values['trade_volume_buys'] = trade['last_size']
        values['trade_volume_sells'] = trade['last_size']

        return values, trade['trade_id']

    def makeFetchDF(self):
        column_list = ['trade_px', 'update_type', 'trades_date_time', 'order_date_time', 'trade_volume', 'trade_volume_buys', 'trade_volume_sells']
        # column_list.append(['trade_volume_buys_minus_asks'])
        # column_list.append(['trade_volume_sells_minus_bids'])
        # column_list.append(['orderbook_market_strength'])
        # column_list.append(['orderbook_market_strength_pct_change'])
        # column_list.append(['trade_px_pct_change'])
        for i in range(1, self.orderbook_range + 1):
            column_list.append('a%s' % i)
            column_list.append('b%s' % i)
            column_list.append('aq%s' % i)
            column_list.append('bq%s' % i)
            # column_list.append('cumsum_aq%s' % i)
            # column_list.append('cumsum_bq%s' % i)
        column_list.append('future_price_%s' % (self.future_price_window))

        dataFrame = pd.DataFrame(columns=column_list)
        return dataFrame

    def initialize_trade_logic(self, exchange=None, new_data_rnn=None, symbol=None, data_window=None):
        coin_symbol, usd = symbol.split('/')
        account_USD = exchange.fetch_balance()['total'][usd]
        account_coin = exchange.fetch_balance()['total'][coin_symbol]
        coin = account_coin
        buy_price = float('{0:2f}'.format(
            new_data_rnn['trade_px'].iloc[-1]))  # Market buy price will be the first ask price
        sell_price = float('{0:2f}'.format(
            new_data_rnn['trade_px'].iloc[-1]))  # Market sell price will be first bid price
        # buy_price = float('{0:2f}'.format(new_data_rnn['b1'].iloc[data_window - 1]))  # Limit buy price will be the first bid price
        # sell_price = float('{0:2f}'.format(new_data_rnn['a1'].iloc[data_window - 1]))  # Limit sell price will be the first ask price
        start_account = float('{0:2f}'.format(account_USD + sell_price * coin))
        print('Start Account = %.2f' % float('{0:2f}'.format(start_account)))
        new_data_rnn = new_data_rnn.reset_index(drop=True)
        # print(new_data_rnn)
        limit_valid = 0
        update_count = -1
        percent_increase = 0.00

        if account_coin >= 0.0001 and account_coin * new_data_rnn['trade_px'].iloc[-1] >= account_USD:
            position = True
            limit_buy_executed = True
            limit_sell_executed = False
        else:
            position = False
            limit_buy_executed = False
            limit_sell_executed = True

        # print ('Previous Order: %s' % trade_exch.fetch_orders(symbol))
        if (trade_exch.fetch_orders(symbol) == []):
            limit_buy_executed = False
            limit_sell_executed = True

        return position, account_USD, start_account, account_coin, coin, buy_price, sell_price, new_data_rnn, coin_symbol, usd, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase

    def market_trade_logic(self, exchange=None, new_data_rnn=None, indicator_data_rnn=None, symbol='BTC/USD',
                           data_window=None, maperiod1=None, maperiod2=None, delay=None,
                           position=None, account_USD=None, account_coin=None, coin=None, buy_price=None,
                           sell_price=None, comm=None, percent=None):
        coin_symbol, usd = symbol.split('/')
        ema1, sma2, ema1_minus_sma2, ema1_minus_trade_px = self.indicators(indicator_data_rnn, 'market', data_window,
                                                                           maperiod1, maperiod2)
        if not position:
            if (ema1_minus_sma2 > 0 and ema1_minus_trade_px < 0):
                position = True
                buy_price = float('{0:2f}'.format(
                    new_data_rnn['trade_px'].iloc[-1]))  # Market buy price will be the first ask price
                coin = float('{0:8f}'.format(percent * account_USD / buy_price))
                # trade_exch.create_order(symbol, type='market', side='buy', amount=coin, price=buy_price)
                account_USD = float('{0:2f}'.format((1 - percent) * account_USD))
                # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))    # Only use this line when making real trades
                print('MARKET BUY CREATED and EXECUTED, Price = %.2f, Account_USD = %.2f' % (buy_price, account_USD))
                time.sleep(delay)


        else:
            if (ema1_minus_sma2 < 0 and ema1_minus_trade_px > 0):
                position = False
                sell_price = float('{0:2f}'.format(
                    new_data_rnn['trade_px'].iloc[-1]))  # Market sell price will be first bid price
                revenue = float('{0:2f}'.format((1 - comm) * coin * (sell_price - buy_price)))
                # trade_exch.create_order(symbol, type='market', side='sell', amount=coin, price=sell_price)
                account_USD = float(
                    '{0:2f}'.format(coin * sell_price + account_USD))  # Use this line for trading fake money
                # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))    # Only use this line when making real trades
                print('MARKET SELL CREATED and EXECUTED, Price = %.2f, Comm = %.4f, Revenue = %.2f, Account = %.2f' % (
                sell_price, comm, revenue, account_USD))
                coin = float('{0:8f}'.format(account_coin))
                time.sleep(delay)

        return position, coin, account_USD, buy_price, sell_price

    def limit_trade_logic(self, exchange=None, new_data_rnn=None, indicator_data_rnn=None, symbol='BTC/USD',
                          data_window=None, maperiod1=None, maperiod2=None, delay=None,
                          position=None, account_USD=None, account_coin=None, coin=None, buy_price=None,
                          sell_price=None, comm=None, percent=None, order_valid=None, limit_valid=None,
                          limit_buy_executed=None,
                          limit_sell_executed=None, start_account=None, update_count=None, percent_increase=None):
        coin_symbol, usd = symbol.split('/')
        ema1, sma2, ema1_minus_sma2, ema1_minus_predicted_px = self.indicators(indicator_data_rnn, 'limit', data_window,
                                                                               maperiod1, maperiod2)
        # a1 = new_data_rnn['a1'][data_window - 1]
        # b1 = new_data_rnn['b1'][data_window - 1]
        # spread = new_data_rnn['spread'][data_window - 1]
        # print('a1 = %.2f, b1 = %.2f, spread = %.2f' % (a1, b1, spread))
        order_status = exchange.fetch_orders(symbol)[0]['status']
        order_side = exchange.fetch_orders(symbol)[0]['side']

        # Make a flash crash segment!!!!!!! if temp_sell_price <= 0.8 * sell_price
        # Make a flash crash segment!!!!!!!
        # Make a flash crash segment!!!!!!!

        if (exchange.fetch_orders(symbol) == []):
            if (ema1_minus_sma2 > 0 and ema1_minus_predicted_px < 0 and limit_sell_executed == True):
                position = True
                limit_buy_executed = False
                limit_sell_executed = False
                buy_price = exchange.fetch_order_book(symbol)['bids'][0][0]
                # if (spread == 0.01):
                #     buy_price = new_data_rnn['b1'][data_window - 1] # Limit buy price will be the first bid price
                # else:
                #     buy_price = new_data_rnn['b1'][data_window - 1] + 0.01   # Limit buy price will supercede first bid price
                coin = float('{0:8f}'.format(percent * account_USD / buy_price))
                exchange.create_order(symbol, type='limit', side='buy', amount=coin, post_only=True,
                                      price=float('{0:2f}'.format(buy_price)))
                account_USD = float('{0:2f}'.format(
                    exchange.fetch_balance()['total'][usd]))  # Only use this line when making real trades
                print('LIMIT BUY CREATED, Price = %.2f, Account_USD = %.2f' % (
                float('{0:2f}'.format(buy_price)), account_USD))
                if (order_status == 'rejected'):
                    print("LIMIT BUY REJECTED DUE TO POST_ONLY = True")
                    position = False
                    limit_buy_executed = False
                    limit_sell_executed = True
                    update_count = -1
                limit_valid = 0
        else:

            if not position:

                if (update_count != -1):
                    if (limit_valid < order_valid):
                        temp_sell_price = exchange.fetch_order_book(symbol)['asks'][0][0]
                        if (order_status == 'closed'):
                            position = False
                            limit_sell_executed = True
                            account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            percent_increase = (100 * (account_USD - start_account) / start_account)
                            print(
                                'LIMIT SELL EXECUTED, Price = %.2f, Comm = %.4f, Account = %.2f, Percent Increase = %.2f' % (
                                float('{0:2f}'.format(sell_price)), comm, account_USD,
                                float('{0:2f}'.format(percent_increase))))
                            limit_valid = 0
                            update_count = -1
                        elif (temp_sell_price < sell_price):
                            position = False
                            limit_buy_executed = False
                            try:
                                exchange.cancel_order(exchange.fetch_orders(symbol)[0]['id'], symbol)
                                if update_count < 3:
                                    print('LIMIT SELL CANCELED but')
                                    sell_price = temp_sell_price
                                    coin = float('{0:8f}'.format(exchange.fetch_balance()['total'][coin_symbol]))
                                    exchange.create_order(symbol, type='limit', side='sell', amount=coin,
                                                          post_only=True,
                                                          price=float('{0:2f}'.format(sell_price)))
                                    print(
                                        'UPDATED LIMIT SELL CREATED, Price = %.2f' % float('{0:2f}'.format(sell_price)))
                                    update_count += 1
                                else:
                                    print('LIMIT SELL CANCELED AND NO LONGER UPDATED UNTIL NEXT INDICATOR SIGNAL')
                                    position = True
                                    limit_buy_executed = True
                                    update_count = -1
                            except:
                                e = sys.exc_info()[0]
                                print('Error: %s' % e)
                                position = True
                                limit_buy_executed = True
                                update_count = -1
                                pass

                            limit_valid = 0
                    elif (limit_valid >= order_valid and order_side == 'sell' and (order_status == 'open' or order_status == 'rejected')):
                        try:
                            # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            # coin = float('{0:8f}'.format(exchange.fetch_balance()['total'][coin_symbol]))
                            limit_buy_executed = True
                            limit_sell_executed = False
                            limit_valid = 0
                            update_count = -1
                            position = True
                            exchange.cancel_order(exchange.fetch_orders(symbol)[0]['id'], symbol)
                            print('LIMIT SELL CANCELED DUE TO DURATION OF ORDER EXCEEDED')

                        except:
                            # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            # coin = float('{0:8f}'.format(exchange.fetch_balance()['total'][coin_symbol]))
                            e = sys.exc_info()[0]
                            print('Error: %s' % e)
                            pass
                elif (ema1_minus_sma2 < 0 and ema1_minus_predicted_px > 0 and limit_sell_executed == True and update_count == -1):
                # elif (ema1_minus_sma2 > 0 and ema1_minus_predicted_px < 0 and limit_sell_executed == True and update_count == -1):
                    position = True
                    limit_buy_executed = False
                    limit_sell_executed = False
                    update_count = 0
                    buy_price = exchange.fetch_order_book(symbol)['bids'][0][0]
                    # account_USD = exchange.fetch_balance()['total'][usd]  # Only use this line when making real trades

                # if (spread == 0.01):
                    #     buy_price = new_data_rnn['b1'][data_window - 1] # Limit buy price will be the first bid price
                    # else:
                    #     buy_price = new_data_rnn['b1'][data_window - 1] + 0.01   # Limit buy price will supercede first bid price
                    coin = float('{0:8f}'.format(percent * account_USD / buy_price))
                    exchange.create_order(symbol, type='limit', side='buy', amount=coin, post_only=True,
                                          price=float('{0:2f}'.format(buy_price)))
                    print('LIMIT BUY CREATED, Price = %.2f, Account_USD = %.2f' % (float('{0:2f}'.format(buy_price)), account_USD))
                    if (order_status == 'rejected'):
                        print("LIMIT BUY REJECTED DUE TO POST_ONLY = True")
                        position = False
                        limit_buy_executed = False
                        limit_sell_executed = True
                        update_count = -1
                    limit_valid = 0
                # print('5')

            else:

                if (update_count != -1):
                    if (limit_valid < order_valid):
                        temp_buy_price = exchange.fetch_order_book(symbol)['bids'][0][0]
                        if (order_status == 'closed'):
                            position = True
                            limit_buy_executed = True
                            account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            print('LIMIT BUY EXECUTED, Price = %.2f, Account_USD = %.2f' % (
                            float('{0:2f}'.format(buy_price)), account_USD))
                            limit_valid = 0
                            update_count = -1
                        elif (temp_buy_price > buy_price):
                            position = True
                            limit_sell_executed = False
                            try:
                                exchange.cancel_order(exchange.fetch_orders(symbol)[0]['id'], symbol)
                                if update_count < 3:
                                    print('LIMIT BUY CANCELED but')
                                    buy_price = temp_buy_price
                                    # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                                    coin = float('{0:8f}'.format(percent * account_USD / buy_price))
                                    exchange.create_order(symbol, type='limit', side='buy', amount=coin, post_only=True,
                                                          price=float('{0:2f}'.format(buy_price)))
                                    print('UPDATED LIMIT BUY CREATED, Price = %.2f' % float('{0:2f}'.format(buy_price)))
                                    update_count += 1
                                else:
                                    print('LIMIT BUY CANCELED AND NO LONGER UPDATED UNTIL NEXT INDICATOR SIGNAL')
                                    position = False
                                    limit_sell_executed = True
                                    update_count = -1
                            except:
                                e = sys.exc_info()[0]
                                print('Error: %s' % e)
                                position=False
                                limit_sell_executed = True
                                update_count = -1
                                pass

                            limit_valid = 0
                    elif (limit_valid >= order_valid and order_side == 'buy' and (order_status == 'open' or order_status == 'rejected')):
                        try:
                            # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            limit_buy_executed = False
                            limit_sell_executed = True
                            limit_valid = 0
                            update_count = -1
                            position = False
                            exchange.cancel_order(exchange.fetch_orders(symbol)[0]['id'], symbol)
                            print('LIMIT BUY CANCELED DUE TO DURATION OF ORDER EXCEEDED')

                        except:
                            # account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                            e = sys.exc_info()[0]
                            print('Error: %s' % e)
                            pass
                elif (ema1_minus_sma2 > 0 and ema1_minus_predicted_px < 0 and limit_buy_executed == True and update_count == -1):
                # elif (ema1_minus_sma2 < 0 and ema1_minus_predicted_px > 0 and limit_buy_executed == True and update_count == -1):
                    sell_price = exchange.fetch_order_book(symbol)['asks'][0][0]
                    coin = exchange.fetch_balance()['total'][coin_symbol]
                    position = False
                    limit_buy_executed = False
                    limit_sell_executed = False
                    update_count = 0
                    # if (spread == 0.01):
                    #     sell_price = new_data_rnn['a1'][data_window - 1]  # Limit sell price will be the first ask price
                    # else:
                    #     buy_price = new_data_rnn['a1'][data_window - 1] - 0.01  # Limit sell price will supercede first ask price
                    exchange.create_order(symbol, type='limit', side='sell', amount=coin, post_only=True,
                                          price=sell_price)
                    account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                    print('LIMIT SELL CREATED, Price = %.2f' % float('{0:2f}'.format(sell_price)))
                    if (order_status == 'rejected'):
                        print("LIMIT SELL REJECTED DUE TO POST_ONLY = True")
                        position = True
                        limit_buy_executed = True
                        limit_sell_executed = False
                        update_count = -1
                    coin = float('{0:8f}'.format(account_coin))
                    limit_valid = 0
            limit_valid += 1
        print('Position: %s, Limit_valid: %s, Update_count: %s, Limit_buy_exec: %s, Limit_sell_exec: %s' % (
        position, limit_valid, update_count, limit_buy_executed, limit_sell_executed))

        return position, coin, account_USD, buy_price, sell_price, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase

    def indicators(self, indicator_data_rnn=None, order=None, data_window=None, maperiod1=None, maperiod2=None):
        if order == 'market':
            ema1 = \
            indicator_data_rnn['trade_px'].iloc[[-maperiod1 - 1, -1]].ewm(span=maperiod1).mean().iloc[-1]
            sma2 = indicator_data_rnn['trade_px'].iloc[[-maperiod2 - 1, -1]].mean()
            ema1_minus_sma2 = ema1 - sma2
            ema1_minus_trade_px = ema1 - indicator_data_rnn['trade_px'].iloc[-1]
            print('Trade_px = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Trade_px = %.2f' %
                  (indicator_data_rnn['trade_px'].iloc[-1], maperiod1, maperiod2, ema1_minus_sma2, maperiod1,
                   ema1_minus_trade_px))
            return ema1, sma2, ema1_minus_sma2, ema1_minus_trade_px

        elif order == 'limit':
            rnn_length = len(indicator_data_rnn['predicted_px'])
            ema1 = \
            indicator_data_rnn['predicted_px'].iloc[[-maperiod1 - 1, -1]].ewm(span=maperiod1).mean().iloc[-1]
            sma2 = indicator_data_rnn['predicted_px'].iloc[-maperiod2 - 1].mean()
            ema1_minus_sma2 = ema1 - sma2
            ema1_minus_predicted_px = ema1 - indicator_data_rnn['predicted_px'].iloc[-1]
            print('Predicted_px = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Predicted_px = %.2f' %
                  (indicator_data_rnn['predicted_px'].iloc[-1], maperiod1, maperiod2, ema1_minus_sma2, maperiod1,
                   ema1_minus_predicted_px))

            # Maybe change indicator scheme depending on historical trade volume
            # Use EMA[5]-SMA[20] for small trading volume
            # Use EMA[5]-SMA[200] for high trading volume = this will elongate the period of oscillation

            return ema1, sma2, ema1_minus_sma2, ema1_minus_predicted_px



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
    num_epochs = 50
    resample_freq = '5s' # Use '20s' when restore=False for optimal training
    update_freq = float(resample_freq[:-1]) / delay # This is used to tell update_data() to add that many new rows of data before deleting the row with oldest data
    normalization_factor = 1.00 # This is a rescaling factor to create a bigger window for training data set
    symbol = 'ETH-USD'
    data_window = int(future_price_window * update_freq) # Use at least 40 to safely avoid len(test_pred_list) = 0
    maperiod1 = 5
    maperiod2 = 20
    comm = 0.0000  # Market trades are 0.25% or 0.0025
    percent = 0.90  # Choose a value between 0 and 1
    order_valid = data_window#60  # Time allowed for a limit trade order to stay opened
    restore = False
    live_trading = False


    print('Loading %s Market...' % symbol)
    trade_exch.load_markets(symbol)  # request markets

    # Create RNN object
    x = model_RNN(orderbook_range, orderbook_window, future_price_window, num_epochs)

    # Read csv file
    # data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/50_exch_gdax_btcusd_snapshot_20180112/exch_gdax_btcusd_snapshot_20180112.csv')
    data_rnn = pd.read_csv('C:/Users/Joseph/Documents/data/exch_gdax_ethusd_snapshot_20180204.csv')
    # data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/50_exch_gdax_ltcusd_snapshot_20180112/exch_gdax_ltcusd_snapshot_20180112.csv')

    # Provide appropriate ckpt file
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/btc_test'
    data_rnn_ckpt = 'C:/Users/Joseph/Documents/data/eth_test'
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/ltc_test'
    # data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/test'


    tf.reset_default_graph()
    #x.ws = WebsocketClient.WebsocketClient(product_id = 'ETH-USD', channel = "ticker")
    #data_rnn, trade_id = x.preloadData(data_window, delay, trade_exch, symbol)

    # Process Data
    x.train_and_predict(restore=restore, live_trading=live_trading, data_rnn=data_rnn, data_rnn_ckpt=data_rnn_ckpt, resample_freq=resample_freq,
                        update_freq=update_freq, normalization_factor=normalization_factor, exchange=trade_exch, symbol=symbol, delay=delay)

    # Testing indicators
    # x.process_data(restore, data_rnn, resample_freq=resample_freq)