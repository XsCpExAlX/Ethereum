from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod1', 10),
        ('maperiod2', 40),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        # self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod1)
        # self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod2)


        # Indicators for the plotting show
        self.ema1 = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.maperiod1)
        self.ema2 = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.maperiod2)

        self.ema1_minus_ema2 = self.ema1 - self.ema2
        # self.ema1_minus_sma1 = self.ema1 - self.sma1

        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
        #                                     subplot=True)
        # bt.indicators.StochasticSlow(self.datas[0])
        # bt.indicators.MACDHisto(self.datas[0])
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
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f, sma[%d] = %.2f, ema[%d] = %.2f' % (self.dataclose[0], self.params.maperiod1, self.sma1[0], self.params.maperiod1, self.ema1[0]))
        self.log('Close, %.2f, ema1[%d] = %.2f, ema2[%d] = %.2f, ema1_minus_ema2 = %.2f' % (self.dataclose[0], self.params.maperiod1, self.ema1[0], self.params.maperiod2, self.ema2[0], self.ema1_minus_ema2[0]))


        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if (self.ema1_minus_ema2[0] > 0 and self.ema1_minus_ema2[-1] < 0):
            # if self.ema1_minus_ema2 < 0 or self.ema1_minus_ema2[-2] > self.ema1_minus_ema2[-1] < self.ema1_minus_ema2[0] or self.ema1_minus_ema2[-1] < 0 < self.ema1_minus_ema2[0]:
            # if (self.ema1_minus_ema2[-2] - self.ema1_minus_ema2[-1] > 1 * (self.ema1_minus_ema2[-1] - self.ema1_minus_ema2[0])):
            # if (self.ema1_minus_ema2[0] > 0 and self.ema1_minus_ema2[-1] < 0):

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(exectype=bt.Order.Limit,
                                      price=self.dataclose[0])

        else:
            if (self.ema1_minus_ema2[0] < 0 and self.ema1_minus_ema2[-1] > 0):
            # if self.ema1_minus_ema2 > 0 or self.ema1_minus_ema2[-1] > 0 > self.ema1_minus_ema2[0]:
            # if (self.ema1_minus_ema2[-2] - self.ema1_minus_ema2[-1] < 1 * (self.ema1_minus_ema2[-1] - self.ema1_minus_ema2[0])):
            # if (self.ema1_minus_ema2[0] < 0 and self.ema1_minus_ema2[-1] > 0):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(exectype=bt.Order.Limit,
                                       price=self.dataclose[0])


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'C:/Users/donut/PycharmProjects/backtrader/backtrader-master/datas/ETHUSD_backtrader.csv')

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        # Do not pass values before this date
        sessionstart=datetime.datetime(2017, 9, 17),
        # Do not pass values before this date
        sessionend=datetime.datetime(2050, 12, 31),
        # Do not pass values after this date
        reverse=False)

    #
    # datetime=0,
    # high=1,
    # low=2,
    # open=3,
    # close=4,
    # volume=5,
    # openinterest=-1

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(10000.00)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.PercentSizer, percents=50)

    # Set the commission3.
    cerebro.broker.setcommission(commission=0.0000)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()