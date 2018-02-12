import time
import os
import datetime as dt
import calendar
from datetime import timedelta
import pandas as pd
import quandl
import requests
from math import floor
import json
from math import *


filename = 'keys.json'
# Get API keys
with open(filename, 'r') as f:
    keys = json.load(f)

QUANDL_KEY = keys['quandlkey']
TIMES_KEY = keys['nytimeskey']

quandl.ApiConfig.api_key = QUANDL_KEY
print("Got keys!")

def main2():
    """ Daily stock close data and news headline data for past year
    Assumes certain directory structure to run.
    """
    number_of_months = 12   # Includes current month
    months = int(raw_input("Enter the number of months to "
                                     "go back for data: "))
    tickers = ['NVDA', 'AMD']   # Ticker symbol list
    ticker_string_list = ','.join(tickers)
    now = dt.datetime.utcnow()
    print("Now: {}".format(now))
    do_once = True

    # Generate the month list for 
    print(ceil(months/12.))
    month_list = [((now.month - (i + 1))%12+1, \
                   now.year + ((now.month - (i + 1))/12)) \
                   for i in range(months)]
    print(month_list)
    for month, year in month_list:
        print(month, year)

        # Quandl API call
        last_day = calendar.monthrange(year, month)[1]
        ticker_data = quandl.get_table('WIKI/PRICES', 
                                        qopts = { 'columns': ['ticker', 'date', 'close'] }, 
                                        ticker = tickers, 
                                        date = { 'gte': dt.date(year, month, 1).strftime('%y-%m-%d'), 
                                                 'lte': dt.date(year, month, last_day).strftime('%y-%m-%d') })
        ticker_data['date'] = [d.date() for d in ticker_data['date']]

        # NY Times API call
        archives_payload = {'api-key' : TIMES_KEY, 
                    'year': year, 
                    'month': month}
        req = requests.get("https://api.nytimes.com/svc/archive/v1/"+str(year)+"/"+str(month)+".json", 
                            params=archives_payload)
        try:
            news_data_json = req.json()
        except ValueError as e:
            print(req)
            raise e
        docs = news_data_json['response']['docs']
        news_docs_df = pd.DataFrame(docs)
        news_docs_df['pub_date'] = [dt.datetime.strptime(str(d[0:10]), '%Y-%m-%d').date() for d in news_docs_df['pub_date']]
        news_docs_df = news_docs_df.rename(columns={'pub_date': 'date'})
        # print(news_docs_df['headline'])
        types = news_docs_df['headline'].apply(lambda x: type(x))
        # print(news_docs_df[types==list]['headline'])
        news_docs_df = news_docs_df[types==dict]
        has_kickers = news_docs_df['headline'].apply(lambda x: 'kicker' in x.keys())
        no_kickers = news_docs_df['headline'].apply(lambda x: 'kicker' not in x.keys())
        has_main = news_docs_df['headline'].apply(lambda x: 'main' in x.keys())
        news_docs_df = news_docs_df[has_main]
        # Headline formatting        
        news_docs_df.loc[has_kickers, 'headline'] = news_docs_df[has_kickers]['headline'].apply(lambda x: x['main']) + ': ' + news_docs_df[has_kickers]['headline'].apply(lambda x: x['kicker'])
        # print(news_docs_df[no_kickers]['headline'].iloc[0].keys())
        # keys = news_docs_df['headline'].apply(lambda x: x.keys())
        # news_docs_df = news_docs_df['main' in keys]
        news_docs_df.loc[no_kickers, 'headline'] = news_docs_df[no_kickers]['headline'].apply(lambda x: x['main'])
        
        # Merge data
        all_data = pd.merge(news_docs_df, ticker_data, on='date', how='left')
        all_data.to_csv('./Data/NYTimesArticlesAndStocks('+ticker_string_list+')_{0}-{1}.csv'.format(month,year), encoding='utf-8')
        print("Writing data to csv...")

    now = dt.datetime.utcnow()
    start_trading_time = dt.datetime.utcnow()
    start_trading_time.replace(hour=13, minute=30, second=0, microsecond=0)
    end_trading_time = dt.datetime.utcnow()
    end_trading_time.replace(hour=20, minute=0, second=0, microsecond=0)
    do_once_after_hours = 1
    ex = dt.datetime(year=now.year, month=now.month, day=now.day, hour=13, minute=30)



# def main():
#     t = RobinhoodTicker(symbol='nvda')
#     df = t.raw_to_dataframe()
#     print_full(df)
#     now = dt.datetime.utcnow()
#     start_trading_time = dt.datetime.utcnow()
#     start_trading_time.replace(hour=13, minute=30, second=0, microsecond=0)
#     end_trading_time = dt.datetime.utcnow()
#     end_trading_time.replace(hour=20, minute=0, second=0, microsecond=0)
#     do_once_after_hours = 1
#     ex = dt.datetime(year=now.year, month=now.month, day=now.day, hour=13, minute=30)
#     while True:
#         packet_time_data = pd.DataFrame()
#         packets = []
#         times = []
#         time_points = [(ex+timedelta(minutes=5)).strftime('%H:%M') for i in range(78)]
#         utc_now = dt.datetime.utcnow()
#         write_csvs = False

#         while utc_now > start_trading_time and utc_now < end_trading_time and \
#         utc_now.weekday not in [5, 6]:
#             utc_now = dt.datetime.utcnow()
#             utc_now_string = utc_now.strftime('%H:%M')
#             if utc_now_string in time_points:
#                 parser = RSSParser()
#                 news_packets = parser.get_headline_packets()
#                 time_data_point = utc_now.replace(second=0, microsecond=0)
#                 packets.append(news_packets)
#                 times.append(time_data_point)
#                 time_points.remove(utc_now_string)
#             do_once_after_hours = 1
#             write_csvs = True

#         if do_once_after_hours > 0:
#             print('After Hours...')
#             if write_csvs:
#                 packet_time_data['Time'] = times
#                 packet_time_data['Headline Packet'] = packets
#                 packet_time_data.to_csv(utc_now.strftime('%m_%d_%y_news.csv'))
#                 df = t.raw_to_dataframe()
#                 df.to_csv('%m_%d_%y_'+t.symbol+'.csv')
#             do_once_after_hours = 0

        


        # print(news_packets)


if __name__ == '__main__':
    main2()

