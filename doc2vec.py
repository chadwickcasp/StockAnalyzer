# Word vecs to doc vecs using gensim doc2vec

import datetime as dt
from math import *
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from headline_analyzer import HeadlineAnalyzer
from operator import itemgetter
from gensim_test import n_most_similar, VocabEncoder, get_vocab
from statistical_analysis import load_word_vectors

months = int(raw_input("Enter the number of months to "
                       "go back for data: "))
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
    # VE = VocabEncoder('./GoogleNews-vectors-negative300.bin')
    # word_vecs_arr = load_word_vectors('word_vectors.npy')
    # print([word_vecs_arr])
    # word_vecs_dict = dict(zip(word_vecs_arr[:,0], word_vecs_arr[:,1]))
    # print(word_vecs_arr)
    # print(word_vecs_arr.shape)
    # sys.exit(0)
    n_analyzed = 1000
    month_list = [((now.month - (i + 1))%12+1, \
                   now.year + ((now.month - (i + 1))/12)) \
                   for i in range(months)]
    all_headlines = pd.DataFrame()
    headlines = []
    dates = []
    percent_changes = []


    # print(VE.model.vocab.keys()[:1000])



    # sys.exit(0)

    for month, year in month_list:
        # month, year = month_year
        month_df = pd.DataFrame()
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

        df = df.reset_index()
        # df['date'] = df['date'].drop_duplicates()
        # df['close'] = df['close'].drop_duplicates()
        # print(df['close'])
        # df['diffs'] = (df['close'] - df['close'].shift(-1))
        # df['diffs'] = df['diffs'].divide(df['close'])*100

        docs = [df.ix[i] for i in df.index]
        print(len(docs))
        tokens_list = []
        for i, article in enumerate(docs):
            text_data = []
            if i%n_analyzed == 0:
                print(i)
                # print(article)
                # print(type(article))
                # print('headline' in article)
                # print(article['headline'])
            if 'headline' in article and not pd.isnull(article['headline']):
                headline_data = article['headline']
                text_data.append(headline_data)
                # tokens = HA.clean_data(headline_data)
            if 'lead_paragraph' in article and not pd.isnull(article['lead_paragraph']):
                paragraph_data = article['lead_paragraph']
                text_data.append(paragraph_data)
            if 'snippet' in article and not pd.isnull(article['snippet']):
                snippet_data = article['snippet']
                text_data.append(snippet_data)

            tokens = max(text_data, key=len)
            # tokens = HA.clean_data(tokens, stem=False)
            # tokens = [HA.strip_and_norm(t) for t in tokens]
            tokens = simple_preprocess(tokens)
            tokens = TaggedDocument(tokens, [article['date']+'-'+str(i)])
            tokens_list.append(tokens)

        # print(tokens_list)
        headlines += tokens_list


    # Make dataframe for doc2vec
    print('Training embedding...')
    model = Doc2Vec(headlines, size=300, window=3, min_count=5, workers=4)
    model.save('NYTimesYearDocVecs.bin')
    print('Done!')

if __name__ == '__main__':
    main()