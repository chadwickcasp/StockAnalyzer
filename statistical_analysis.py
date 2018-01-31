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

for month, year in month_list[0]:
    print(month, year)
    df = pd.read_csv('./Data/NYTimesArticlesAndStocks('+ticker_string_list+')_{0}-{1}.csv'.format(month,year))
    df = df.drop_duplicates('_id')
    df = df.reset_index()
    data_shape = df.shape
    length = data_shape[0]
    docs = [df.ix[i] for i in range(length)]
    bigrams_dict = {}
    trigrams_dict = {}
    grams_dict = {}
    print(df.columns)
    # for i, article in enumerate(docs):
