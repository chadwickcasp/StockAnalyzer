import datetime as dt
from math import *
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from headline_analyzer import HeadlineAnalyzer
from operator import itemgetter
from gensim_test import n_most_similar, VocabEncoder, get_vocab
from statistical_analysis import load_word_vectors
import time

def main():
    model = Doc2Vec.load('NYTimesYearDocVecs.bin')
    model.delete_temporary_training_data(keep_doctags_vectors=True, 
                                         keep_inference=True)
    # print(model)
    # print(help(model))
    # print(model.docvecs[1])
    # print(type(model.docvecs[1]))
    # print(model.iter)
    # print(dir(model))
    # for obj in dir(model):
    #     print(obj)
    #     print(getattr(model, obj))
    #     print('')
    # print(len(model.syn1neg))
    print(len(model.docvecs))
    # print(model.docvecs)
    # for obj in dir(model.docvecs):
    #     print(obj)
    #     print(getattr(model.docvecs, obj))
    #     print('')
    # print(model.docvecs.keys())
    print(len(model.docvecs.offset2doctag))
    print(model.docvecs.offset2doctag[:1000])

if __name__ == '__main__':
    start = time.time()
    main()
    print('Elapsed time for load: {}'.format(time.time() - start))
