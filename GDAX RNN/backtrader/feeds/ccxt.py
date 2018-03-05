from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import deque
from datetime import datetime
from time import sleep

import backtrader as bt
from backtrader.feed import DataBase

import ccxt

class CCXT(DataBase):
    # Supported granularities
    _GRANULARITIES = {
        (bt.TimeFrame.Minutes, 1): '1m',
        (bt.TimeFrame.Minutes, 3): '3m',
        (bt.TimeFrame.Minutes, 5): '5m',
        (bt.TimeFrame.Minutes, 15): '15m',
        (bt.TimeFrame.Minutes, 30): '30m',
        (bt.TimeFrame.Minutes, 60): '1h',
        (bt.TimeFrame.Minutes, 90): '90m',
        (bt.TimeFrame.Minutes, 120): '2h',
        (bt.TimeFrame.Minutes, 240): '4h',
        (bt.TimeFrame.Minutes, 360): '6h',
        (bt.TimeFrame.Minutes, 480): '8h',
        (bt.TimeFrame.Minutes, 720): '12h',
        (bt.TimeFrame.Days, 1): '1d',
        (bt.TimeFrame.Days, 3): '3d',
        (bt.TimeFrame.Weeks, 1): '1w',
        (bt.TimeFrame.Weeks, 2): '2w',
        (bt.TimeFrame.Months, 1): '1M',
        (bt.TimeFrame.Months, 3): '3M',
        (bt.TimeFrame.Months, 6): '6M',
        (bt.TimeFrame.Years, 1): '1y',
    }

    def __init__(self, exchange, symbol, ohlcv_limit=10):
        self.exchange = getattr(ccxt, exchange)()
        self.symbol = symbol
        self.ohlcv_limit = ohlcv_limit

        self._data = deque() # data queue for price data
        self._last_id = None # last processed data id (trade id or timestamp for ohlcv)

    def _load(self):
        sleep(self.exchange.rateLimit / 1000) # time.sleep wants seconds
        self._timeframe=1
        if self._timeframe == bt.TimeFrame.Ticks:
            return self._load_ticks()
        elif self.exchange.hasFetchOHLCV:
            granularity = self._GRANULARITIES.get((self._timeframe, self._compression))
            if granularity is None:
                raise ValueError("'%s' exchange doesn't support fetching OHLCV data for "
                                 "time frame %s, comression %s" % \
                                 (self.exchange.name, bt.TimeFrame.getname(self._timeframe),
                                  self._compression))
            else:
                return self._load_ohlcv(granularity)
        else:
            raise NotImplementedError("'%s' exchange doesn't support fetching OHLCV data" % \
                                      self.exchange.name)

    def _load_ohlcv(self, granularity):
        for ohlcv in self.exchange.fetch_ohlcv(self.symbol, granularity, limit=self.ohlcv_limit):
            tstamp = ohlcv[0]
            if tstamp > self._last_id:
                self._data.append(ohlcv)
                self._last_id = tstamp

        try:
            ohlcv = self._data.popleft()
        except IndexError:
            return  # no bars in the queue

        tstamp, open_, high, low, close, volume = ohlcv

        dtime = datetime.utcfromtimestamp(tstamp // 1000)

        self.lines.datetime[0] = bt.date2num(dtime)
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume

        print("loaded bar time: %s, open: %s, high: %s, low: %s, close: %s, volume: %s" % \
              (dtime.strftime('%Y-%m-%d %H:%M:%S'), open_, high, low, close, volume))

        return True

    def _load_ticks(self):
        if self._last_id is None:
            # first time get the latest trade only
            trades = [self.exchange.fetch_trades(self.symbol)[-1]]
            self._last_id = 0
        else:
            trades = self.exchange.fetch_trades(self.symbol)

        for trade in trades:
            trade_id = trade['info']['trade_id']
            if trade_id > self._last_id:
                tradetime = trade['info']['time']
                if (tradetime.find('.') > 0):
                    trade_time = datetime.strptime(tradetime, '%Y-%m-%dT%H:%M:%S.%fZ')
                else:
                    trade_time = datetime.strptime(tradetime, '%Y-%m-%dT%H:%M:%SZ')
                self._data.append((trade_time, float(trade['info']['price']), float(trade['info']['size'])))
                self._last_id = trade_id

        try:
            trade = self._data.popleft()
        except IndexError:
            return # no trades in the queue

        trade_time, price, size = trade

        self.lines.datetime[0] = bt.date2num(trade_time)
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