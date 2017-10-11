import pandas as pd

#TODO: parallelize for better performance
def parseData(readData, outData):
    df = pd.read_csv(readData)

    ### sort by date just in case the data isn't
    df['order_date_time']=pd.to_datetime(df.order_date_time)
    df=df.sort_values('order_date_time')
    odt=df['order_date_time']

    #initialize some objects
    isNewMinute = False
    i=0
    current_minute = odt[0].minute
    trade_px,trade_volume,b1,b2,b3,b4,b5,a1,a2,a3,a4,a5,bq1,bq2,bq3,bq4,bq5,aq1,aq2,aq3,aq4,aq5,order_date_time = ([] for m in range(23)) # skipped from the original data: id, trades_date_time, update_type
    trade_px_,trade_volume_,b1_,b2_,b3_,b4_,b5_,a1_,a2_,a3_,a4_,a5_,bq1_,bq2_,bq3_,bq4_,bq5_,aq1_,aq2_,aq3_,aq4_,aq5_,order_date_time_ = (0 for k in range(23)) # skipped from the original data: id, trades_date_time, update_type

    # go through each row, and compile the data by the minute
    # add all the quantities within the minute, and get the average values for each column
    for j in range(0,len(odt)):
        current_time = odt[j]
        isNewMinute = current_time.minute != current_minute
        if isNewMinute:
            current_minute = (current_minute+1)%60
            order_date_time.append('%s-%s-%s %s:%s:00' %(order_date_time_.year, order_date_time_.month, order_date_time_.day, order_date_time_.hour, order_date_time_.minute))
            order_date_time_ = odt[j]
            trade_px.append(trade_px_/i)
            trade_px_=df['trade_px'][j]
            trade_volume.append(trade_volume_)
            trade_volume_=df['trade_volume'][j]
            a1.append(a1_/i)
            a1_ =df['a1'][j]
            a2.append(a2_/i)
            a2_=df['a2'][j]
            a3.append(a3_/i)
            a3_=df['a3'][j]
            a4.append(a4_/i)
            a4_=df['a4'][j]
            a5.append(a5_/i)
            a5_=df['a5'][j]
            aq1.append(aq1_/i)
            aq1_ =df['aq1'][j]
            aq2.append(aq2_/i)
            aq2_=df['aq2'][j]
            aq3.append(aq3_/i)
            aq3_=df['aq3'][j]
            aq4.append(aq4_/i)
            aq4_=df['aq4'][j]
            aq5.append(aq5_/i)
            aq5_=df['aq5'][j]
            b1.append(b1_/i)
            b1_ =df['b1'][j]
            b2.append(b2_/i)
            b2_=df['b2'][j]
            b3.append(b3_/i)
            b3_=df['b3'][j]
            b4.append(b4_/i)
            b4_=df['b4'][j]
            b5.append(b5_/i)
            b5_=df['b5'][j]
            bq1.append(bq1_/i)
            bq1_ =df['bq1'][j]
            bq2.append(bq2_/i)
            bq2_=df['bq2'][j]
            bq3.append(bq3_/i)
            bq3_=df['bq3'][j]
            bq4.append(bq4_/i)
            bq4_=df['bq4'][j]
            bq5.append(bq5_/i)
            bq5_=df['bq5'][j]
            i=1
        else:
            order_date_time_ = odt[j]
            trade_px_ += df['trade_px'][j]
            trade_volume_ += df['trade_volume'][j]
            a1_ += df['a1'][j]
            a2_ += df['a2'][j]
            a3_ += df['a3'][j]
            a4_ += df['a4'][j]
            a5_ += df['a5'][j]
            aq1_ += df['aq1'][j]
            aq2_ += df['aq2'][j]
            aq3_ += df['aq3'][j]
            aq4_ += df['aq4'][j]
            aq5_ += df['aq5'][j]
            b1_ += df['b1'][j]
            b2_ += df['b2'][j]
            b3_ += df['b3'][j]
            b4_ += df['b4'][j]
            b5_ += df['b5'][j]
            bq1_ += df['bq1'][j]
            bq2_ += df['bq2'][j]
            bq3_ += df['bq3'][j]
            bq4_ += df['bq4'][j]
            bq5_ += df['bq5'][j]
            i+=1

    # set up the dataframe object to prepare converting to .csv
    columns = ['order_date_minute','trade_px','trade_volume','a1','aq1','a2','aq2','a3','aq3','a4','aq4','a5','aq5','b1','bq1','b2','bq2','b3','bq3','b4','bq4','b5','bq5']
    d = {'order_date_minute':order_date_time, 'trade_px':trade_px,'trade_volume':trade_volume,'a1':a1,'a2':a2,'a3':a3,'a4':a4,'a5':a5,'aq1':aq1,'aq2':aq2,'aq3':aq3,'aq4':aq4,'aq5':aq5,'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5,'bq1':bq1,'bq2':bq2,'bq3':bq3,'bq4':bq4,'bq5':bq5}
    df1 = pd.DataFrame(d, columns=columns)

    # export the averages to .CSV
    df1.to_csv(outData,index=False)


if __name__ == "__main__":
    readData = 'C:/Users/Joe/Documents/workspace/exch_gdax_btcusd_snapshot_20170906.csv'
    outData = 'C:/Users/Joe/Documents/workspace/predictions1_bitcoin.csv'
    parseData(readData, outData)
