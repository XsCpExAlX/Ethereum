from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

import datetime
import backtrader.feeds as btfeed

class RNN (bt.feeds.GenericCSVData):

  params = (
    ('fromdate', datetime.datetime(2017, 1, 1)),
    ('todate', datetime.datetime(2018, 12, 31)),
    ('nullvalue', 0.0),
    # ('dtformat', ('%Y-%m-%d %H:%M:%S')),    # Make sure to also change dtformat in csvgeneric.py
    ('dtformat', ('%Y-%m-%d %H:%M:%S.%f')),    # Make sure to also change dtformat in csvgeneric.py
    # ('dtformat', ('%Y%m%d %H:%M:%S.%f')),
    ('tmformat', ('%H.%M:%S')),

    ('datetime', 0),
    ('time', -1),
    ('open', 1),
    ('high', 2),
    ('low', 3),
    ('close', 4),
    ('RNN', -1),
    ('openinterest', -1),

    # ('DATETIME', 0),
    # ('time', -1),
    # ('BTC_USD', 1),
    # ('BTC_USD_GDAX', 2),
    # ('RHsell', 3),
    # ('RHbuy', 4),

)


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod1', 50),  # maperiod1=5 if no repeats (5 * 10)
        ('maperiod2', 200), # maperiod2=200 if no repeats (20 * 10)
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datalow = self.datas[0].low

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        # self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod1)
        # self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod2)


        # Indicators for the plotting show
        self.ema1 = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.params.maperiod1)
        self.sma1 = bt.indicators.MovingAverageSimple(self.dataclose, period=self.params.maperiod1)
        self.sma2 = bt.indicators.MovingAverageSimple(self.dataclose, period=self.params.maperiod2)

        self.SD = bt.indicators.StdDev(self.dataclose, period=self.params.maperiod1)

        self.ema1_minus_sma2 = self.ema1 - self.sma2
        # self.ema1_minus_sma1 = self.ema1 - self.sma1

        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
        #                                     subplot=True)
        # self.stochastic = bt.indicators.StochasticSlow(self.datas[0])
        # self.MACD = bt.indicators.MACDHisto(self.datas[0])
        # rsi = bt.indicators.RSI(self.datas[0])
        # bt.indicators.SmoothedMovingAverage(rsi, period=10)
        # bt.indicators.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed = len(self)

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        SD_multiplier=2

        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f, sma[%d] = %.2f, ema[%d] = %.2f' % (self.dataclose[0], self.params.maperiod1, self.sma1[0], self.params.maperiod1, self.ema1[0]))
        # self.log('Close, %.2f, ema1[%d] = %.2f, sma2[%d] = %.2f' % (self.dataclose[0], self.params.maperiod1, self.ema1[0], self.params.maperiod2, self.sma2[0]))
        # self.log('Low, %.2f, Close, %.2f, ema1[%d] = %.2f, sma2[%d] = %.2f' % (self.datalow[0], self.dataclose[0], self.params.maperiod1, self.ema1[0], self.params.maperiod2, self.sma2[0]))
        self.log('Close, %.2f, SMA1[%d] = %.2f, StdDev[%d] = %.2f' % (self.dataclose[0], self.params.maperiod1, self.sma1[0], self.params.maperiod1, self.SD[0]))


        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            # if (self.ema1_minus_sma2[0] > 0 and self.ema1_minus_sma2[-1] < 0):
            if (self.ema1_minus_sma2[0] > 0 and self.ema1[0] < self.dataclose[0]):
            # if (self.sma1[0] - self.dataclose[0] >= SD_multiplier*self.SD[0]):

                # BUY, BUY, BUY!!! (with all possible default parameters)
                # self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.log('BUY CREATE, %.2f' % (self.dataclose[0] - 20))

                # Keep track of the created order to avoid a 2nd order
                # self.order = self.buy(exectype=bt.Order.Market,
                #                       price=self.dataclose[0])
                self.order = self.buy(exectype=bt.Order.Limit,
                                      price=self.dataclose[0] - 20,
                                      valid=datetime.timedelta(minutes=5))

        else:

            # if (self.ema1_minus_sma2[0] < 0 and self.ema1_minus_sma2[-1] > 0):
            if (self.ema1_minus_sma2[0] < 0 and self.ema1[0] > self.dataclose[0]):
            # if (self.dataclose[0] - self.sma1[0] >= SD_multiplier*self.SD[0]):


                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % (self.dataclose[0] + 20))
                # self.log('SELL CREATE, %.2f' % self.datalow[0])

                # Keep track of the created order to avoid a 2nd order
                # self.order = self.sell(exectype=bt.Order.Market,
                #                        price=self.dataclose[0])
                self.order = self.sell(exectype=bt.Order.Limit,
                                      price=self.dataclose[0] + 20,
                                      valid=datetime.timedelta(minutes=5))

if __name__ == '__main__':

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/bitflyer_BTCUSD_TRADEDATA_120517_bt_datetime.csv')
    datapath = os.path.join(modpath, 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/BTCUSD_composite3-sep.csv')


    # Create a Data Feed
    data = RNN(
            dataname=datapath,
            # Do not pass values before this date
            sessionstart=datetime.datetime(2017, 9, 17),
            # Do not pass values before this date
            sessionend=datetime.datetime(2050, 12, 31),
            # Do not pass values after this date
            reverse=False)



    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(10000.00)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.PercentSizer, percents=80)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)


    # Set the commission (limit order is 0.0000)
    cerebro.broker.setcommission(commission=0.000)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()

