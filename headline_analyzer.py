import nltk
import re
import regex
import string
import pytz
# nltk.download()
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from difflib import ndiff

class HeadlineAnalyzer():
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.porter = PorterStemmer()

    def remove_stopwords(self, token_list):
        return [t for t in token_list if t not in stopwords.words('english')]

    def remove_non_joiners(self, token_list):
        new_token_list = []

        for token in token_list:
            new_token = self.regex.sub(u'', token)
            # new_token = regex.sub('[\p{P}\p{Sm}]+', '', token)
            new_token = unicodedata.normalize('NFKD', new_token).encode('ascii', 'ignore')
            # Handle if not solely punctuation (words in addition to punctuation)
            if not new_token == u'':
                punctuation = [c[-1] for c in ndiff(token, new_token)]
                if len(token) > len(new_token) and \
                  ('-' in punctuation or '_' in punctuation):
                    new_token_list.append(token)
                    continue
                new_token_list.append(new_token)
        # table = string.maketrans("","")
        # new_token_list = [t.translate(table, string.punctuation) for t in token_list]
        return new_token_list

    def remove_punctuation(self, token_list):
        new_token_list = []

        for token in token_list:
            new_token = self.regex.sub(u'', token)
            # new_token = regex.sub('[\p{P}\p{Sm}]+', '', token)
            new_token = unicodedata.normalize('NFKD', new_token).encode('ascii', 'ignore')
            # Handle if not solely punctuation (words in addition to punctuation)
            if not new_token == u'':
                punctuation = [c[-1] for c in ndiff(token, new_token)]
                if len(token) > len(new_token) and \
                  '-' in punctuation:
                    new_token_list.append(token)
                    continue
                new_token_list.append(new_token)
        # table = string.maketrans("","")
        # new_token_list = [t.translate(table, string.punctuation) for t in token_list]
        return new_token_list

    def remove_html_tags(self, string):
        p = re.compile(r'<.*?>')
        new_token = p.sub('', string)
        return new_token

    # Tokenize and remove html, stopwords, punctuation from headline
    def clean_data(self, headline):
        # Decode anything that looks utf-8
        ex_ = headline.decode('utf-8')
        # print(ex_)
        ex_ = self.remove_html_tags(ex_)
        ex_ = word_tokenize(ex_)
        # print(ex_)
        ex_ = self.remove_stopwords(ex_)
        # print(ex_)
        ex_ = self.remove_punctuation(ex_)
        # print(ex_)
        # Remove endings from words to make the problem simpler
        ex_ = [self.porter.stem(x) if x[0] in string.ascii_lowercase else x for x in ex_]
        ex_ = [stem.encode('utf-8') for stem in ex_]
        return ex_
