from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import time
import ccxt
from ccxt_sandbox_api import sandbox_api_key, sandbox_secret_key, sandbox_passphrase
from ccxt_real_api import real_api_key, real_secret_key


# Import the RNN packages
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ccxt_rnn_class import model_RNN


# Instantiate sandbox gdax api to execute practice trades
sandbox = ccxt.gdax({'apiKey': sandbox_api_key,
                     'secret': sandbox_secret_key,
                     'password': sandbox_passphrase,
                     'nonce': ccxt.gdax.seconds,
                     'verbose': False}) # If verbose is True, log HTTP requests
sandbox.urls['api'] = 'https://api-public.sandbox.gdax.com'


# Instantiate real gdax api to get live data feed
gdax = ccxt.gdax({'apiKey': real_api_key,
                  'secret': real_secret_key,
                  'password': real_passphrase,
                  'nonce': ccxt.gdax.seconds,
                  'verbose': False}) # If verbose is True, log HTTP requests
gdax.urls['api'] = 'https://api.gdax.com'


delay = 2 # delay in seconds
sandbox.load_markets()
gdax.load_markets() # request markets
# orderbook = gdax.fetch_order_book ('BTC/USD')
# bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
# ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
# spread = (ask - bid) if (bid and ask) else None
# print (gdax.id, 'market price', {'bid': bid, 'ask': ask, 'spread': spread})
# print(orderbook)

print('Total USD: %f, Total BTC: %f' % (gdax.fetch_balance()['total']['USD'], gdax.fetch_balance()['total']['BTC']))

# if gdax.hasFetchOHLCV:
#     time.sleep(gdax.rateLimit / 1000) # time.sleep wants seconds
#     # print(gdax.fetch_ohlcv('BTC/USD', '1m')) # one minute
#     buy1 = gdax.create_order(market='BTC/USD', type='limit', side='buy', amount=0.06, price=500.00)
#     # gdax.create_order(market='BTC/USD', type='market', side='buy', amount=0.04)
#     sell1 = gdax.create_order(market='BTC/USD', type='limit', side='sell', amount=0.03, price=10000.00)
#     # gdax.create_order(market='BTC/USD', type='market', side='sell', amount=0.02)
#     gdax.cancel_order(id=sell1['id'])
#     print(gdax.fetch_balance())
#     print(gdax.fetch_order_book('BTC/USD', {'depth': 10}))