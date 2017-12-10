from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from datetime import datetime
import urllib3
import os
import sys
import time
import ccxt
from bitflyer_real_api import access_key, secret_key
from gdax_real_api import real_api_key, real_secret_key, real_passphrase


# Import the RNN packages
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ccxt_rnn_class import model_RNN


# Instantiate real bitflyer api to get live data feed
exch = ccxt.bitflyer({'apiKey': access_key,
                      'secret': secret_key,
                      'nonce': ccxt.bitflyer.seconds,
                      'verbose': False}) # If verbose is True, log HTTP requests
exch.urls['api'] = 'https://api.bitflyer.com'


# # Instantiate real gdax api to get live data feed
# # Currently running gdax.py with orderbook level 1
# exch = ccxt.gdax({'apiKey': real_api_key,
#                   'secret': real_secret_key,
#                   'password': real_passphrase,
#                   'nonce': ccxt.gdax.seconds,
#                   'verbose': False}) # If verbose is True, log HTTP requests
# exch.urls['api'] = 'https://api.gdax.com'


# start = datetime.now()
print('Loading Market')
exch.load_markets() # request markets
# print(datetime.now() - start)


# start = datetime.now()
x = model_RNN(order_book_range=1, order_book_window=1, future_price_window=20, future_ma_window=20, num_epochs=1)
data_window = 200
maperiod1 = 50
maperiod2 = 200
count = 0
delay = 5

new_data_rnn = x.preloadData(data_window, delay, exch)
print('Preloading Data with window size: %s' % data_window)
# ema1 = new_data_rnn['trade_px'][::-1].ewm(span=maperiod1).mean()[0]
ema1 = new_data_rnn['trade_px'][[data_window - maperiod1 - 1, data_window - 1]].ewm(span=maperiod1).mean()[data_window - 1]
sma2 = new_data_rnn['trade_px'].mean()
ema1_minus_sma2 = ema1 - sma2
ema1_minus_trade_px = ema1 - new_data_rnn['trade_px'][data_window - 1]
print('Loop %s, Trade_px = %.2f, EMA[%s] = %.2f, SMA[%s] = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Trade_px = %.2f' %
      (count, new_data_rnn['trade_px'][data_window - 1], maperiod1, ema1, maperiod2, sma2, maperiod1, maperiod2, ema1_minus_sma2, maperiod1, ema1_minus_trade_px))
count += 1
time.sleep(delay)
# print(datetime.now() - start)


# Keep track of order and position status
account_USD = float('{0:2f}'.format(exch.fetch_balance()['total']['USD']))
account_BTC = float('{0:8f}'.format(exch.fetch_balance()['total']['BTC']))
coin = float('{0:8f}'.format(account_BTC))
# print(account)
buy_price = float('{0:2f}'.format(new_data_rnn['trade_px'][data_window - 1]))
sell_price = float('{0:2f}'.format(new_data_rnn['trade_px'][data_window - 1]))
comm = 0.00
percent = 0.95   # Percent of account that will be used to execure buy order. Choose values between 0.00 and 1.00
profit = float('{0:2f}'.format((1 - comm) * (sell_price - buy_price)))
if account_BTC >= 0.001 and account_BTC * new_data_rnn['trade_px'][data_window - 1] >= account_USD:
    position = True
else:
    position = False


count = 1
while True:
    # print('Loop #%s' % count)
    # start = datetime.now()
    new_data_rnn = new_data_rnn.drop(new_data_rnn.head(1).index)  # take out the leftmost
    new_data_rnn = pd.concat([new_data_rnn, x.fetchExchangeData(exch)])  # add it to ends
    new_data_rnn = new_data_rnn.reset_index(drop=True)
    # ema1 = new_data_rnn['trade_px'][::-1].ewm(span=maperiod1).mean()[0]
    ema1 = new_data_rnn['trade_px'][[data_window - maperiod1 - 1, data_window - 1]].ewm(span=maperiod1).mean()[data_window - 1]
    sma2 = new_data_rnn['trade_px'].mean()
    ema1_minus_sma2 = ema1 - sma2
    ema1_minus_trade_px = ema1 - new_data_rnn['trade_px'][data_window - 1]
    print('Loop %s, Trade_px = %.2f, EMA[%s] = %.2f, SMA[%s] = %.2f, EMA[%s]_minus_SMA[%s] = %.2f, EMA[%s]_minus_Trade_px = %.2f' %
          (count, new_data_rnn['trade_px'][data_window - 1], maperiod1, ema1, maperiod2, sma2, maperiod1, maperiod2, ema1_minus_sma2, maperiod1, ema1_minus_trade_px))

    if not position:
        if (ema1_minus_sma2 > 0 and ema1_minus_trade_px < 0):
            position = True
            buy_price = float('{0:2f}'.format(new_data_rnn['trade_px'][data_window - 1]))
            coin = float('{0:8f}'.format(percent * account_USD / buy_price))
            account_USD = float('{0:2f}'.format((1 - percent) * account_USD))
            print('BUY EXECUTED, Price = %.2f' % buy_price)
            exch.create_order('BTC/USD', type='market', side='buy', amount=coin)
    else:
        if (ema1_minus_sma2 < 0 and ema1_minus_trade_px > 0):
            position = False
            sell_price = float('{0:2f}'.format(new_data_rnn['trade_px'][data_window - 1]))
            account_USD = float('{0:2f}'.format(coin * sell_price + account_USD))
            profit = float('{0:2f}'.format(coin * (sell_price - buy_price)))
            print('SELL EXECUTED, Price = %.2f, Comm = %.4f, Profit = %.2f, Account = %s' % (sell_price, comm, profit, account_USD))
            exch.create_order('BTC/USD', type='market', side='sell', amount=coin)
            coin = float('{0:8f}'.format(account_BTC))


    # print(datetime.now() - start)
    count += 1
    time.sleep(delay)


# if gdax.hasFetchOHLCV:
#     time.sleep(gdax.rateLimit / 1000) # time.sleep wants seconds
#     # print(gdax.fetch_ohlcv('BTC/USD', '1m')) # one minute
#     buy1 = gdax.create_order(market='BTC/USD', type='limit', side='buy', amount=0.06, price=500.00)
#     # gdax.create_order(market='BTC/USD', type='market', side='buy', amount=0.04)
#     sell1 = gdax.create_order(market='BTC/USD', type='limit', side='sell', amount=0.03, price=10000.00)
#     # gdax.create_order(market='BTC/USD', type='market', side='sell', amount=0.02)
#     gdax.cancel_order(id=sell1['id'])
#     print(gdax.fetch_account())
#     print(gdax.fetch_order_book('BTC/USD', {'depth': 10}))