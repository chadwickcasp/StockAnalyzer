import gensim
from gensim.models.keyedvectors import KeyedVectors
import time
import pandas as pd
import numpy as np
import re
import string
from headline_analyzer import HeadlineAnalyzer


LIMIT = 1000000
# while True:
#     print("Every second.")
#     time.sleep(1)


def elem_removed(lst, el):
    lst.remove(el)
    return lst

def n_most_similar(n, word, model):
    word_lst = []
    n_similar = []
    print('Running N most similar...')
    for _ in range(n):
        most_similar = model.most_similar(w, word_lst)
        n_similar.append(most_similar)
        word_lst.remove(most_similar)
    return n_similar


def get_vocab(model):
    word_lst = []
    try:
        print('Getting Vocab...')
        word_lst = model.wv.vocab.keys()
        print('Done getting Vocab.')
    except AttributeError as e:
        word_lst = model.vocab.keys()
    except AttributeError as e:
        print('Can\'t get vocab from model.')
    return word_lst


def n_least_edits(n, word, vocab):
    import editdistance
    n_similar = []
    print('Running N most similar...')
    # print(word_lst.keys())
    for _ in range(n):
        mindist = float('inf')
        for w in vocab:
            dist = editdistance.eval(word, w)
            if dist < mindist:
                most_similar = w
                mindist = dist
        n_similar.append((most_similar, mindist))
        vocab.remove(most_similar)
    return n_similar

def main():
    vocab_df = pd.read_csv('vocab.csv', index_col = 0)
    vocab_df = vocab_df.sort_values(by='Frequency')

    print("Loading the Google Word2Vec Model...")
    w2v_path = './GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(w2v_path, 
                                              binary=True,
                                              limit=LIMIT)  
    word2vec_vocab = get_vocab(model)
    print("Loaded.")
    HA = HeadlineAnalyzer()

    words = list(vocab_df['Word'])
    word_vecs = []
    n_words = len(words)
    non_matches = []
    for i, w in enumerate(words):
        if i % 100 == 0:
            print('On {0}/{1} word'.format(i, n_words))
        if w:
            try:
                vec = model[w]
                pair = [w, vec]
                # print(pair)
                pair_arr = np.array(pair, dtype=object)
                word_vecs.append(pair_arr)
            except KeyError as e:
                print('Word {} not in model vocab.'.format(w))
                non_matches.append(w)
                continue

    word_vec_pairs = np.array(word_vecs)

    print('Looking for alternatives for non-matches...')
    for i, w in enumerate(non_matches):
        if i % 100 == 0:
            print('On {0}/{1} word'.format(i, len(non_matches)))
        if w:
            vocab = word_vec_pairs[:, 0]
            # five_most_similar = model.most_similar(positive=[w], topn=5)
            five_least_edits = n_least_edits(5, w, word2vec_vocab)
            print('{0} most similar to {1}'.format(five_least_edits, w))
            for tup in five_least_edits:
                word = tup[0]
                # Check for different circumstances
                w_lower, word_lower = string.lower(w), string.lower(word)
                w_lower = re.sub('-', '_', w_lower)
                word_lower = re.sub('-', '_', word_lower)
                word_lower = HA.remove_non_joiners([word_lower])[0]
                word_lower.decode('utf-8')
                print(word_lower)
                if w_lower in word_lower or word_lower in w_lower:
                    print('Found a match!')
                    print('Matched {0} to {1}.'.format(w_lower, word_lower))
                    vec = model[word]
                    pair = [[w, vec]]
                    pair_arr = np.array(pair, dtype=object)
                    np.append(word_vec_pairs, pair_arr, axis=0)
                    non_matches.remove(w)
                    break
                # time.sleep(10)
            # except Exception as e:
            #     print('Couldn\'t find most similar words.')
            #     # raise(e)
    print(non_matches)
    diff = abs(len(word_vecs) - len(words))
    print('Number of word missing form word2vec vocab: {}'.format(diff))
    with open('word_vectors.npy', 'wb') as f:
        np.save(f, word_vec_pairs)
    # df.save_csv('word_vecs.csv')

if __name__ == '__main__':
    main()
