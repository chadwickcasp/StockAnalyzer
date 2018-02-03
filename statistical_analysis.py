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

def main():
    word_vecs_arr = load_word_vectors('word_vectors.npy')
    print(word_vecs_arr)
    print(word_vecs_arr.shape)
    sys.exit(0)
    n_analyzed = 30
    month_list = [((now.month + i-1)%12+1, \
                    int(floor((now.month+i)/13)+now.year-1)) for i \
                    in range(number_of_months+1)]
    for ticker in tickers:
        overall_diffs = pd.DataFrame()
        for month, year in month_list:
            print(month, year)
            try:
                file = './Data/NYTimesArticlesAndStocks('+\
                       ticker_string_list+\
                       ')_{0}-{1}.csv'.format(month,year)
                df = pd.read_csv(file)
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
            # print(df.columns)
            diffs = pd.DataFrame()
            diffs['date'] = df['date'].drop_duplicates()
            diffs['close'] = df['close'].drop_duplicates()
            overall_diffs = overall_diffs.append(diffs)

        # print(overall_diffs)
        # print(overall_diffs.columns)
        overall_diffs = overall_diffs.sort_values('date')
        overall_diffs['diffs'] = (overall_diffs['close'] - overall_diffs['close'].shift(-1))
        overall_diffs['diffs'] = overall_diffs['diffs'].divide(overall_diffs['close'])*100
        # print(overall_diffs)
        positive_diffs = overall_diffs[overall_diffs['diffs'] > 0]
        num_positive = len(positive_diffs)
        negative_diffs = overall_diffs[overall_diffs['diffs'] < 0]
        num_negative = len(negative_diffs)
        zero_diffs = overall_diffs[overall_diffs['diffs'] == 0]
        num_zero = len(zero_diffs)
        print("Ratio of positive to "
              "negative labels: {0}:{1}:{2}".format(num_positive,
                                                    num_negative,
                                                    num_zero))
        # plt.figure()
        overall_diffs.hist(column='diffs', alpha=0.7, bins=50)
        plt.title("Frequency of Day-to-day Percent Changes of {}".format(ticker))
    plt.show()

def load_word_vectors(filename):
    arr=np.array([])
    with open(filename, 'rb') as f:
        arr = np.load(f)
    # except IOError as e:
    #     print("Couldn't load the word vectors...")
    #     print("Check file name and try again")
    return arr

def map_headline_to_vector_list(headline, word_vecs_arr):
    '''Maps words in a head line to a list of word2vec vectors
    '''
    if 'headline' in article and not pd.isnull(article['headline']):
        headline_data = article['headline']
        tokens = HA.clean_data(headline_data)
        tokens_list.append(tokens)
    if 'lead_paragraph' in article and not pd.isnull(article['lead_paragraph']):
        tokens_list.append(HA.clean_data(article['lead_paragraph']))
    if 'snippet' in article and not pd.isnull(article['snippet']):
        tokens_list.append(HA.clean_data(article['snippet']))

    # Pick the largest chunk of data among the different types
    tokens = max(tokens_list, key=len)
    word_vecs_dict = dict(zip(word_vecs_arr[::2], word_vecs_arr[1::2]))
    hl_vecs = [word_vecs_dict[token] for token in tokens]


def avg_headline(headline):
    '''Averages all of the word vectors over a headline.
    '''


def word_vector_analysis(word_vecs_arr, headline_df):
    '''Finds things like the average word for a certain market move (positive
    or negative in a given month
    '''

if __name__ == '__main__':
    main()

    # for i, article in enumerate(docs):
