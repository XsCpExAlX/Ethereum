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

from bitflyer_real_api import access_key, secret_key
from gdax_real_api import real_api_key, real_secret_key, real_passphrase


class VariableHolder:  # variable holder
    # num_classes, truncated_backprop_length, num_features, last_state, last_label, prediction, batchX_placeholder, batchY_placeholder
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

    def __init__(self, order_book_range, order_book_window, future_price_window, future_ma_window, num_epochs):
        self.order_book_range = order_book_range
        self.order_book_window = order_book_window
        self.future_price_window = future_price_window
        self.future_ma_window = future_ma_window
        self.num_epochs = num_epochs

    def process_data(self, data_rnn, restore):
        # print('nrows of raw data= %s' % len(data_rnn.index))

        # Convert 'trades_date_time' and  'order_date_time' from object to datetime object
        data_rnn['trades_date_time'] = pd.to_datetime(data_rnn['trades_date_time'])

        # Add any columns that are needed

        data_rnn['spread'] = data_rnn['a1'] - data_rnn['b1']

        for i in range(1, self.order_book_range + 1):
            data_rnn['aqq%s' % i] = data_rnn[['a%s' % i, 'aq%s' % i]].product(axis=1)
            data_rnn['bqq%s' % i] = data_rnn[['b%s' % i, 'bq%s' % i]].product(axis=1)

        for i in range(1, self.future_price_window + 1):
            data_rnn['future_price_%s' % i] = data_rnn['trade_px']
            data_rnn['future_ma_%s' % self.future_ma_window] = data_rnn['trade_px'][::-1].rolling(
                window=self.future_price_window).mean()[::-1]

        # Resample data by setting index to 'trades_date_time' to avoid repeat
        data_rnn = data_rnn[data_rnn.columns.difference([
                                                            'order_date_time'])]  # .difference() method removes any columns and automatically reorders columns alphanumerically
        data_rnn = data_rnn.resample('S', on='trades_date_time').mean().interpolate(method='linear')
        # print('nrows with resample = %s' % len(data_rnn.index))

        # if 'row_num' not in data_rnn.columns: #TODO: add rownum for new incoming data
        data_rnn.insert(0, 'row_num',
                        range(len(data_rnn.index)))  # This column is used as a surrogate for the row number

        # Normalize data
        PriceRange = data_rnn['trade_px'].max() - data_rnn['trade_px'].min()
        PriceMean = data_rnn['trade_px'].mean()
        data_rnn_norm = (data_rnn - data_rnn.mean()) / (data_rnn.max() - data_rnn.min())

        return PriceRange, PriceMean, data_rnn_norm, data_rnn

    def train_and_predict(self, restore=False, data_rnn=None, data_rnn_ckpt=None, exchange=None, symbol=None,
                          delay=None, data_window=None, maperiod1=None, maperiod2=None, comm=None, percent=None,
                          order_valid=None):
        # tf.reset_default_graph()
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

        PriceRange, PriceMean, data_rnn_norm, data_rnn_processed = self.process_data(data_rnn, restore)

        rnn_column_list = self.get_rnn_column_list()

        # RNN Placeholders
        batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_features],
                                            name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, truncated_backprop_length, num_classes],
                                            name='target_ph')

        # RNN Train-Test Split
        test_first_idx = 0  # needed for train
        if restore:
            data_rnn_test = data_rnn_norm
            # print('nrows of testing = %s' % len(data_rnn_test.index))

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
            # data_rnn_train = data_rnn_norm[data_rnn['row_num'] < (test_first_idx-2400)]
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

        # Add saver variable to save and restore all variables from trained model
        saver = tf.train.Saver()

        # Initialize and run session
        with tf.Session() as sess:
            if restore:
                saver.restore(sess, data_rnn_ckpt)
                # predicted_price = 0
                # data_rnn, ema1, sma2, ema1_minus_sma2, ema1_minus_trade_px = self.indicators(data_rnn, data_window, maperiod1, maperiod2)
                position, account_USD, start_account, account_coin, coin, buy_price, sell_price, data_rnn, coin_symbol, \
                usd, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase = self.initialize_trade_logic(
                    exchange, data_rnn, symbol, data_window)

                while True:
                    start_time = timer()
                    # if update_count > 0:
                    #     account_USD = float('{0:2f}'.format(exchange.fetch_balance()['total'][usd]))
                    #     account_coin = float('{0:8f}'.format(exchange.fetch_balance()['total'][coin_symbol]))
                    updated, updated_data_rnn = self.updateData(exchange, data_rnn, symbol)
                    # print('updated: ', updated_data_rnn)

                    if updated:
                        data_rnn = updated_data_rnn
                        # print('data: ', data_rnn)
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
                            _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction],
                                                                           feed_dict=feed)
                            test_pred_list.append(test_pred[-1][0])  # The last one

                        actual_price = data_rnn.tail(1)['trade_px']
                        predicted_price = test_pred_list[-1] * PriceRange + PriceMean
                        test_pred_list[:] = [(x * PriceRange) + PriceMean for x in test_pred_list]
                        data_rnn_predicted = pd.DataFrame(test_pred_list, columns=['predicted_px'])
                        # print('data_rnn: %s, data_rnn_predicted: %s' % (len(data_rnn['trade_px']), len(data_rnn_predicted)))
                        # print("Actual price: %s, Predicted price: %s" % (actual_price.item(), float('{0:2f}'.format(predicted_price))))
                        # position, coin, account_USD, buy_price, sell_price = self.market_trade_logic(exchange, data_rnn, data_rnn, symbol, data_window, maperiod1, maperiod2, delay, position, account_USD, account_coin, coin, buy_price, sell_price, comm, percent)
                        position, coin, account_USD, buy_price, sell_price, limit_valid, limit_buy_executed, limit_sell_executed, update_count, percent_increase = self.limit_trade_logic(
                            exchange, data_rnn, data_rnn_predicted, symbol, data_window, maperiod1, maperiod2, delay,
                            position, account_USD,
                            account_coin, coin, buy_price, sell_price, comm, percent, order_valid=order_valid,
                            limit_valid=limit_valid,
                            limit_buy_executed=limit_buy_executed, limit_sell_executed=limit_sell_executed,
                            start_account=start_account, update_count=update_count, percent_increase=percent_increase)
                        print("time taken for iteration: ", timer() - start_time)
                        print("")
                        # self.plot_predictions(test_pred_list, yTest, PriceRange, PriceMean)
                        # time.sleep(delay)
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
        predict = pd.DataFrame(test_pred_list, columns=['predicted_px'])
        real = pd.DataFrame(yTest, columns=['trade_px'])
        real_vs_predict = predict.join(real)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        #    print(real_vs_predict)

        # # Print out graphs for loss and Price vs Prediction
        # plt.title('Loss')
        # plt.scatter(x=np.arange(0, len(loss_list)), y=loss_list)
        # plt.xlabel('epochs')
        # plt.ylabel('loss')

        plt.figure(figsize=(15, 6))
        plt.plot(yTest, label='trade_px', color='blue')
        plt.plot(test_pred_list, label='predicted_px', color='red')
        plt.title('trade_px vs predicted_px')
        plt.legend(loc='upper left')
        plt.show()

    def predict(self, sess, data_rnn, num_classes, truncated_backprop_length, num_features, last_state, last_label,
                prediction, batchX_placeholder, batchY_placeholder):
        # truncated_backprop_length=5
        # num_features=20
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

    def updateData(self, exchange, data_rnn, symbol):
        new_data_rnn, new_trade_id = self.fetchExchangeData(exchange, symbol)
        if new_trade_id != self.last_trade_id:
            # input("update")
            data_rnn = data_rnn.drop(data_rnn.head(1).index)
            data_rnn = pd.concat([data_rnn, new_data_rnn])
            data_rnn = data_rnn.reset_index(drop=True)
            # input(data_rnn)
            return True, data_rnn

        return False, data_rnn

    def preloadData(self, iteration=None, interval=None, exchange=None, symbol=None):
        print("Starting preload")
        data = self.makeFetchDF()

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

        self.last_trade_id = trade_id
        return data.sort_values(by='trades_date_time'), trade_id

    def fetchExchangeData(self, exchange, symbol):
        data = self.makeFetchDF()

        trades = exchange.fetch_trades(symbol)
        trade = trades[0]

        orderbook = exchange.fetch_order_book(symbol)
        asks = orderbook['asks']
        bids = orderbook['bids']

        values = {'trade_px': trade['price'], 'update_type': trade['side'], 'trades_date_time': trade['datetime']}
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

    def initialize_trade_logic(self, exchange=None, new_data_rnn=None, symbol=None, data_window=None):
        coin_symbol, usd = symbol.split('/')
        account_USD = exchange.fetch_balance()['total'][usd]
        account_coin = exchange.fetch_balance()['total'][coin_symbol]
        coin = account_coin
        buy_price = float('{0:2f}'.format(
            new_data_rnn['trade_px'].iloc[data_window - 1]))  # Market buy price will be the first ask price
        sell_price = float('{0:2f}'.format(
            new_data_rnn['trade_px'].iloc[data_window - 1]))  # Market sell price will be first bid price
        # buy_price = float('{0:2f}'.format(new_data_rnn['b1'].iloc[data_window - 1]))  # Limit buy price will be the first bid price
        # sell_price = float('{0:2f}'.format(new_data_rnn['a1'].iloc[data_window - 1]))  # Limit sell price will be the first ask price
        start_account = float('{0:2f}'.format(account_USD + sell_price * coin))
        print('Start Account = %.2f' % float('{0:2f}'.format(start_account)))
        new_data_rnn = new_data_rnn.reset_index(drop=True)
        # print(new_data_rnn)
        limit_valid = 0
        update_count = -1
        percent_increase = 0.00

        if account_coin >= 0.0001 and account_coin * new_data_rnn['trade_px'][data_window - 1] >= account_USD:
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
                    new_data_rnn['trade_px'][data_window - 1]))  # Market buy price will be the first ask price
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
                    new_data_rnn['trade_px'][data_window - 1]))  # Market sell price will be first bid price
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
            indicator_data_rnn['trade_px'][[data_window - maperiod1 - 1, data_window - 1]].ewm(span=maperiod1).mean()[
                data_window - 1]
            sma2 = indicator_data_rnn['trade_px'][[data_window - maperiod2 - 1, data_window - 1]].mean()
            ema1_minus_sma2 = ema1 - sma2
            ema1_minus_trade_px = ema1 - indicator_data_rnn['trade_px'][data_window - 1]
            print('Trade_px = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Trade_px = %.2f' %
                  (indicator_data_rnn['trade_px'][data_window - 1], maperiod1, maperiod2, ema1_minus_sma2, maperiod1,
                   ema1_minus_trade_px))
            return ema1, sma2, ema1_minus_sma2, ema1_minus_trade_px

        elif order == 'limit':
            rnn_length = len(indicator_data_rnn['predicted_px'])
            ema1 = \
            indicator_data_rnn['predicted_px'][[rnn_length - maperiod1 - 1, rnn_length - 1]].ewm(span=maperiod1).mean()[
                rnn_length - 1]
            sma2 = indicator_data_rnn['predicted_px'][rnn_length - maperiod2 - 1].mean()
            ema1_minus_sma2 = ema1 - sma2
            ema1_minus_predicted_px = ema1 - indicator_data_rnn['predicted_px'][rnn_length - 1]
            print('Predicted_px = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Predicted_px = %.2f' %
                  (indicator_data_rnn['predicted_px'][rnn_length - 1], maperiod1, maperiod2, ema1_minus_sma2, maperiod1,
                   ema1_minus_predicted_px))

            # Maybe change indicator scheme depending on historical trade volume
            # Use EMA[5]-SMA[20] for small trading volume
            # Use EMA[5]-SMA[200] for high trading volume = this will elongate the period of oscillation

            return ema1, sma2, ema1_minus_sma2, ema1_minus_predicted_px


if __name__ == '__main__':  # TODO: modularize train_and_predict (take out load and rnnCELL), Fetch data by listening to websocket.

    # Instantiate real gdax api to get live data feed
    # Currently running gdax.py with orderbook level 1
    trade_exch = ccxt.gdax({'apiKey': real_api_key,
                            'secret': real_secret_key,
                            'password': real_passphrase,
                            'nonce': ccxt.gdax.seconds,
                            'verbose': False})  # If verbose is True, log HTTP requests
    trade_exch.urls['api'] = 'https://api.gdax.com'
    # orderbook_exch = ccxt.gdax()

    # # Instantiate real bitflyer api to get live data feed
    # trade_exch = ccxt.bitflyer({'apiKey': access_key,
    #                             'secret': secret_key,
    #                             'nonce': ccxt.bitflyer.seconds,
    #                             'verbose': False}) # If verbose is True, log HTTP requests
    # trade_exch.urls['api'] = 'https://api.bitflyer.com'
    # # orderbook_exch = ccxt.bitflyer()

    # new_data_rnn = pd.read_csv('C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/exch_gdax_ethusd_snapshot_20170913.csv', nrows=10000)
    data_rnn_ckpt = 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/rnn_saved_models/50_exch_gdax_ethusd_snapshot_20180112/50_exch_gdax_ethusd_snapshot_20180112'
    x = model_RNN(order_book_range=5, order_book_window=1, future_price_window=20, future_ma_window=20, num_epochs=50)

    print('Loading Market...')
    trade_exch.load_markets()  # request markets

    symbol = 'ETH/USD'
    data_window = 35
    maperiod1 = 5
    maperiod2 = 20
    comm = 0.0000  # Market trades are 0.25% or 0.0025
    percent = 0.90  # Choose a value between 0 and 1
    delay = 0.3  # Time between each pull request
    order_valid = 60  # Time allowed for a limit trade order to stay opened

    new_data_rnn, trade_id = x.preloadData(data_window, delay, exchange=trade_exch, symbol=symbol)
    # new_data_rnn.to_csv("C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/preload_data.csv")  # for testing. We can save the data from preload and just reuse that for testing so we don't have to wait every execution.
    # new_data_rnn = pd.read_csv("C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/preload_data.csv")

    tf.reset_default_graph()

    x.train_and_predict(restore=True, data_rnn=new_data_rnn, data_rnn_ckpt=data_rnn_ckpt, exchange=trade_exch,
                        symbol=symbol,
                        delay=delay, data_window=data_window, maperiod1=maperiod1, maperiod2=maperiod2, comm=comm,
                        percent=percent, order_valid=order_valid)
    # x.train_and_predict(restore=False, data_rnn=new_data_rnn, data_rnn_ckpt=data_rnn_ckpt, exchange=trade_exch, symbol=symbol, delay=delay)