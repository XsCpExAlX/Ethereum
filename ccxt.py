from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import deque
from datetime import datetime
from time import sleep

import backtrader as bt
from backtrader.feed import DataBase

import ccxt

class CCXT(DataBase):
    def __init__(self, exchange, symbol):
        DataBase.__init__(self)
        print(symbol)
        self.exchange = getattr(ccxt, exchange)()
        self.symbol = symbol
        self.trades = deque()
        self.last_trade_id = None

    def _load(self):
        sleep(self.exchange.rateLimit / 1000) # time.sleep wants seconds
        if self.last_trade_id is None:
            # first time get the latest trade only
            self.last_trade_id = 1
            trades = [self.exchange.fetch_trades(self.symbol)[-1]]
        else:
            trades = self.exchange.fetch_trades(self.symbol)    
            
        for trade in trades:
            trade_id = trade['info']['trade_id']

            if trade_id > self.last_trade_id:
                trade_time = datetime.strptime(trade['info']['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
                self.trades.append((trade_time, float(trade['info']['price']), float(trade['info']['size'])))
                self.last_trade = trade_id

        try:
            trade = self.trades.popleft()
        except IndexError:
            return False # no trades in the queue

        trade_time, price, size = trade
        dt = bt.date2num(trade_time)
        #if dt <= self.lines.datetime[-1]:
        #    return False # cannot deliver earlier than already delivered

        self.lines.datetime[0] = dt

        self.lines.open[0] = price
        self.lines.high[0] = price
        self.lines.low[0] = price
        self.lines.close[0] = price
        self.lines.volume[0] = size

        print("loaded tick time: %s, price: %s, size: %s" % (trade_time, price, size))

        return True

    def haslivedata(self):
        return True  # must be overriden for those that can

    def islive(self):
        return True