import time
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import calendar
from datetime import timedelta
import pandas as pd
import quandl
import requests
import math
from math import floor
import json
import ast
from headline_analyzer import HeadlineAnalyzer
from operator import itemgetter

number_of_months = 12   # Includes current month
tickers = ['NVDA', 'AMD']   # Ticker symbol list
ticker_string_list = ','.join(tickers)
now = dt.datetime.utcnow()
do_once=True
HA = HeadlineAnalyzer()
freq_dict = {}

def n_grams(tokens, n):
    """Returns all of the n grams in a text document
    """
    return zip(*[tokens[i:] for i in range(n)])

n_analyzed = 30
month_list = [((now.month + i-1)%12+1, int(floor((now.month+i)/13)+now.year-1)) for i \
              in range(number_of_months+1)]
for ticker in tickers:
    overall_diffs = pd.DataFrame()
    for month, year in month_list:
        print(month, year)
        try:
            df = pd.read_csv('./Data/NYTimesArticlesAndStocks('+ticker_string_list+')_{0}-{1}.csv'.format(month,year))
        except IOError as e:
            print("Month data not found in directory...")
            print("Check list and try again")
            break
        # df = df.drop_duplicates('_id')
        df = df.reset_index()
        # print(df[u'ticker'].drop_duplicates())
        df = df[df['ticker'] == ticker]
        # print(df)
        # data_shape = df.shape
        # length = data_shape[0]
        # # docs = [df.ix[i] for i in range(length)]
        # bigrams_dict = {}
        # trigrams_dict = {}
        # grams_dict = {}
        print(df.columns)
        diffs = pd.DataFrame()
        diffs['date'] = df['date'].drop_duplicates()
        diffs['close'] = df['close'].drop_duplicates()
        overall_diffs = overall_diffs.append(diffs)

    print(overall_diffs)
    print(overall_diffs.columns)
    overall_diffs = overall_diffs.sort_values('date')
    overall_diffs['diffs'] = (overall_diffs['close'] - overall_diffs['close'].shift(-1))
    overall_diffs['diffs'] = overall_diffs['diffs'].divide(overall_diffs['close'])*100
    print(overall_diffs)
    plt.figure()
    overall_diffs.hist(column='diffs', alpha=0.7, bins=50)
    plt.title("Frequency of Day-to-day Percent Changes of {}".format(ticker))
plt.show()
    # for i, article in enumerate(docs):
