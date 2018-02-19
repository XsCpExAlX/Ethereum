
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#1 is buy for update type


def findPeaks(data):
    #df = pd.read_csv("C:/Users/Joseph/Documents/data/exch_gdax_btcusd_snapshot_20180204.csv", nrows=100000)
    df = data

    df['trade_px']=df[["b1", "a1"]].mean(axis=1)
    
    df['sign_vol']=df['trade_volume'].loc[df['update_type'] == 1]
    df['sign_vol']=df['sign_vol'].combine_first(-df['trade_volume'].loc[df['update_type'] == 2])
    
    df['aqsum5']= df['aq1']+df['aq2']+df['aq3']+df['aq4']+df['aq5']
    df['bqsum5']= df['bq1']+df['bq2']+df['bq3']+df['aq4']+df['aq5']
    
    df['abq']= df['aq1']-df['bq1']
    df['abq5']= df['aqsum5']-df['bqsum5']
    
    tdt=pd.to_datetime(df.trades_date_time).to_frame()
    tdt["Spread"] = df["a1"]-df["b1"]
    tdt['trade_px']=df['trade_px']
    
    tdt["a1"] = df["a1"]
    tdt["b1"] = df["b1"]
    tdt["aq1"] = df["aq1"]
    tdt["bq1"] = df["bq1"]
    
    tdt["aqsum5"] = df["aqsum5"]
    tdt["bqsum5"] = df["bqsum5"]
    tdt['abq']=df['abq']
    tdt['abq5']=df['abq5']
    tdt['abratio']= df['aq1']/(df['aq1']+df['bq1'])
    #tdt['ratio5']=pd.rolling_mean(tdt['abratio'], 5)
    tdt['ratio5']=tdt['abratio'].rolling(window= 5, center=False).mean()
    tdt['sign_vol']=df['sign_vol']


    tdt["trade_volume"]=df["trade_volume"]
    
    tdt=tdt.drop(tdt.index[:5])
    
    
    tdt.index = pd.to_datetime(tdt['trades_date_time'])
    tdt = tdt.resample('s').mean()
    #tdt=tdt.interpolate(method='linear')
    
    """
    plt.figure(figsize=(25,7))
    plt.plot(tdt['trade_px'], label = 'trade_px')
    plt.legend(loc='upper left')
    plt.show()
    """
    
    tdt['diff']=tdt['trade_px'].diff(periods=1).round(8)
    
    
    tdt['norepeat']=tdt['diff'].loc[tdt['diff'].shift(-1) != 0]
    diff = tdt[tdt['diff'] != 0]
    nans = np.isnan(diff)
    #norepeat = norepeat[np.isfinite(tdt['norepeat'])]
    start = tdt[tdt['norepeat'] == 0]
    start=start.drop('diff', axis=1)
    start=start.drop('norepeat', axis=1)
    start['zero']=0
    
    
    #with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
       # print(tdt)
    
    
    start['future_px'] = start['trade_px'].shift(-1)
    #print(start.columns.values)
    
    
    start = start[['trade_px', 'future_px', 'abq','abq5','abratio', 'ratio5','sign_vol','trade_volume']]
    
    """
    plt.figure(figsize=(25,7))
    plt.plot(start['trade_px'], label = 'starts only',linestyle='',marker='o',markersize=3,color='red')
    plt.plot(tdt['trade_px'], label = 'trade_px')
    
    plt.legend(loc='upper left')
    plt.show()
    """
    
    
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(start)
    
    
    start.reset_index(inplace=True)
    
    
    start['price_change']=start['future_px']-start['trade_px']
    start['before price_change']=1
    
    return start
    
    #start.to_csv('C:\\Users\\Joseph\\Documents\\BTC_Slopes.csv',index=False)
    
    
    
    
    
    """
    #start['god_mode']=start['price_change' > 0].sum()
    start['god_mode'] = np.where(start['price_change'] > 0, 0, start['price_change'])
    start['god_mode'] = start['god_mode'].sum()
    """
    
    
    """
    plt.scatter( start['abq'], start['price_change'], label = 'abq1')
    plt.scatter(start['abq5'], start['price_change'], label = 'abq5')
    plt.legend(loc='upper left')
    plt.suptitle('Order book sum vs. Price Change', fontsize=20)
    plt.ylabel('Price Change', fontsize=18)
    plt.xlabel('Order Book sum', fontsize=16)
    plt.show()
    
    
    plt.scatter(start['trade_volume'], start['price_change'])
    plt.suptitle('Order book sum vs. Price Change', fontsize=20)
    plt.ylabel('Price Change', fontsize=18)
    plt.xlabel('Volume', fontsize=16)
    plt.show()
    
    
    plt.scatter(start['abratio'], start['price_change'] )
    plt.suptitle('Price Change vs. ab ratio', fontsize=20)
    plt.ylabel('Price Change', fontsize=18)
    plt.xlabel('ratio', fontsize=16)
    plt.show()
    
    
    
    plt.scatter(start['ratio5'], start['price_change'],color='blue' )
    plt.scatter(start['abratio'], start['price_change'], color='red' )
    plt.suptitle('Price Change vs. ratio10', fontsize=20)
    plt.ylabel('Price Change', fontsize=18)
    plt.xlabel('ratio5', fontsize=16)
    plt.show()
    
    
    plt.scatter(start['sign_vol'], start['price_change'] )
    plt.suptitle('Price Change vs. ab ratio', fontsize=20)
    plt.ylabel('Price Change', fontsize=18)
    plt.xlabel('sign_vol', fontsize=16)
    plt.ylim( (-10, 10) )
    plt.xlim(-1, 1)
    plt.show()
    """
    
    
    
