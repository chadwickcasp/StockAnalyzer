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
month_list = [((now.month + i-1)%12+1, int(floor((now.month+i)/13)+now.year-1)) for i in range(number_of_months+1)]

for month, year in month_list:
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
    for i, article in enumerate(docs):
        tokens_list = []

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

        # Indicate progress of data parsing
        if i % 20 == 0:
            print('{0} / {1}'.format(i, len(docs)))

        # Get n-grams wanted
        bigram_tokens = n_grams(tokens, 2)
        trigram_tokens = n_grams(tokens, 3)

        # Frequency dictionary incrementing
        for t in bigram_tokens:
            if t not in bigrams_dict:
                bigrams_dict[t] = 1
            else:
                bigrams_dict[t] += 1
        for t in trigram_tokens:
            if t not in trigrams_dict:
                trigrams_dict[t] = 1
            else:
                trigrams_dict[t] += 1
        for t in tokens:
            if t not in grams_dict:
                grams_dict[t] = 1
            else:
                grams_dict[t] += 1
        for t in tokens:
            if t not in freq_dict:
                freq_dict[t] = 1
            else:
                freq_dict[t] += 1

    # Get top n-grams for plotting
    sorted_bigrams = sorted(bigrams_dict.items(), key=itemgetter(1), reverse=True)
    sorted_trigrams = sorted(trigrams_dict.items(), key=itemgetter(1), reverse=True)
    top_bigram_freq = zip(*sorted_bigrams[:n_analyzed])[1]
    top_trigram_freq = zip(*sorted_trigrams[:n_analyzed])[1]
    top_bigrams = zip(*sorted_bigrams[:n_analyzed])[0]
    top_trigrams = zip(*sorted_trigrams[:n_analyzed])[0]
    top_bigrams = [' '.join([word for word in g]).decode('utf-8') 
                   for g in top_bigrams]
    top_trigrams = [' '.join([word for word in g]).decode('utf-8')
                    for g in top_trigrams]

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10,8))

    y_pos = np.arange(len(top_bigrams))
    ax.barh(y_pos, top_bigram_freq, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_bigrams)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Frequency')
    ax.set_title('Top {2} Bigrams of articles from {0}-{1}'.format(month, year, n_analyzed))
    plt.tight_layout()

    plt.rcdefaults()
    fig2, ax2 = plt.subplots(figsize=(10,8))

    y_pos = np.arange(len(top_trigrams))
    ax2.barh(y_pos, top_trigram_freq, align='center',
            color='green', ecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_trigrams)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Frequency')
    ax2.set_title('Top {2} Trigrams of articles from {0}-{1}'.format(month, year, n_analyzed))
    plt.tight_layout()

    # plt.show()

    grams_df = pd.DataFrame(grams_dict.items(), columns=['Word', 'Frequency'])
    print(grams_df.columns)
    grams_df.to_csv('vocab_{}.csv'.format(' '.join(map(str, (month, year)))))

freq_df = pd.DataFrame(freq_dict.items(), columns=['Word', 'Frequency'])
print(freq_df.columns)
freq_df.to_csv('vocab.csv')


# if do_once:
#     article = df.ix[0]
#     for piece in zip(df.columns,article):
#         print(piece)
#         print('\n')
#     tokens_list = []
#     if 'headline' in article.keys():
#         tokens_list.append(HA.clean_data(article['headline']))
#     if 'lead_paragraph' in article.keys():
#         tokens_list.append(HA.clean_data(article['lead_paragraph']))
#     if 'snippet' in article.keys():
#         tokens_list.append(HA.clean_data(article['snippet']))
#     # print(headline_tokens, lead_paragraph_tokens, snippet_tokens)
#     tokens = max(tokens_list, key=len)
#     bigram_tokens = n_grams(tokens, 2)
#     trigram_tokens = n_grams(tokens, 3)
#     print(bigram_tokens)
#     print(trigram_tokens)
#     do_once = False