import gensim
from gensim.models.keyedvectors import KeyedVectors
import time
import pandas as pd
import numpy as np
import re
import string
import pickle
from headline_analyzer import HeadlineAnalyzer


LIMIT = 1000000
pkl_path = 'non_matches.pkl'
no_known_matches_pkl = 'no_known_matches.pkl'
# while True:
#     print("Every second.")
#     time.sleep(1)


def elem_removed(lst, el):
    lst.remove(el)
    return lst

def n_most_similar(n, word_vec, model):
    # word_lst = []
    # n_similar = []
    # print('Running N most similar...')
    # for _ in range(n):
    #     most_similar = model.most_similar(w, word_lst)
    #     n_similar.append(most_similar)
    #     word_lst.remove(most_similar)
    distances = model.distances(word_vec, get_vocab(model))
    print("Got distances!")
    # print(distances)
    # n_similar = model.similar_by_vector(word_vec, n)
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

class VocabEncoder():

    def __init__(self, model_path):
        """Encode a vocab using a model (for now just gensim is supported)
        
        Args:
            vocab: The vocab of your data. Can be JSON data, a file path to
            JSON data or a pandas dataframe.
            model: Gensim model, which acts as a dictionary mapping word to
            encoding (e.g. a vector using word2vec)

        Usage:
            ve = VocabEncoder(vocab, model)
            # Encode creates an encoding matching words to encoding value.
            # It returns it and saves the encoding as a numpy matrix in a
            # .npy file
            ve.encode(options)
        
        """
        print("Loading the Google Word2Vec Model...")
        w2v_path = model_path
        self.model = KeyedVectors.load_word2vec_format(w2v_path, 
                                                  binary=True,
                                                  limit=LIMIT)  
        print("Loaded.")
        sims_init = self.model.init_sims()
        print("Vectors normalized.")

def contains_mult_words(word):
    hyphen = re.compile('[a-zA-Z]-[a-zA-Z]')
    underscore = re.compile('[a-zA-Z]_[a-zA-Z]')
    f_slash = re.compile('[a-zA-Z]/[a-zA-Z]')
    b_slash = re.compile('[a-zA-Z]\[a-zA-Z]')
    if hyphen.search(word):
        return True, '-'
    if underscore.search(word):
        return True, '_'
    if f_slash.search(word):
        return True, '/'
    if b_slash.search(word):
        return True, '\\'
    else:
        return False, None

def encode_utf8_or_nothing(string):
    try:
        string = string.encode('utf-8')
    except UnicodeDecodeError as e:
        print("Word not unicode. Don't do anything")
    return string

def find_word_encoding(w, model):
    try:
        vec = model[w]
        pair = [w, vec]
        pair_arr = np.array(pair, dtype=object)
        return True, pair_arr
    except KeyError as e:
        print('Word {} not in model vocab.'.format(w))
        vec = None
        pair = [w, vec]
        pair_arr = np.array(pair, dtype=object)
        return False, pair_arr



def main():
    vocab_df = pd.read_csv('vocab.csv', index_col = 0)
    vocab_df = vocab_df.sort_values(by='Frequency')

    VE = VocabEncoder('./GoogleNews-vectors-negative300.bin')
    model = VE.model
    word2vec_vocab = get_vocab(model)
    HA = HeadlineAnalyzer()

    words = list(vocab_df['Word'])
    word_vecs = []
    n_words = len(words)
    non_matches = []
    no_known_matches = []

    # First run through the vocab to find encodings
    for i, w in enumerate(words):
        if i % 100 == 0:
            print('On {0}/{1} word'.format(i, n_words))
        if w:
            found, result = find_word_encoding(w, model)
            if found:
                word_vecs.append(result)
            else:
                non_matches.append(w)

    # Second run through non-matches to match multiword non-matches to 
    # encodings
    to_remove = []
    for i, nonmatch in enumerate(non_matches):
        if i % 100 == 0:
            print('On {0}/{1} word'.format(i, len(non_matches)))
        if nonmatch:
            true, character = contains_mult_words(nonmatch)
            if true:
                words = nonmatch.split(character)
                split_success = False
                for word in words:
                    found, result = find_word_encoding(w, model)
                    if found:
                        word_vecs.append(result)
                        split_success = True
                    else:
                        non_matches.append(result)
                if split_success:
                    to_remove.append(nonmatch)

    # Remove words that had a match
    if to_remove:
        for w in to_remove:
            non_matches.remove(w)

    word_vec_pairs = np.array(word_vecs)

    # Pickle load non-matches at this point
    # In case the following code errors
    try:
        with open(pkl_path, 'rb') as file:
            prev_non_matches = pickle.load(file)
            non_matches = non_matches + prev_non_matches
            non_matches = list(set(non_matches))
            print("Pickled file read into non_matches.")
    except IOError as e:
        print("Couldn't open pickled non-matches.")
        print("Continuing...")

    # Looks through non-matches 
    print('Looking for alternatives for non-matches...')
    nm = non_matches[:]
    for i, nonmatch in enumerate(nm):
        if i % 100 == 0:
            print('On {0}/{1} word'.format(i, len(non_matches)))
        if nonmatch:
            five_least_edits = n_least_edits(5, nonmatch, word2vec_vocab)
            print('{0} most similar to {1}'.format(five_least_edits, nonmatch))
            nm_normed = HA.strip_and_norm(nonmatch.decode('utf-8'))
            non_matches.remove(nonmatch)

            # Look through the words with minimum edit distance from word
            for tup in five_least_edits:
                pot_match = tup[0]
                pm_normed = HA.strip_and_norm(pot_match)

                # If either word is part of the other, consider a match
                if nm_normed in pm_normed or pm_normed in nm_normed:
                    print('Found a match!')
                    _pot_match = encode_utf8_or_nothing(pot_match)
                    _nonmatch = encode_utf8_or_nothing(nonmatch)
                    print('Matched {0} to {1}.'.format(_nonmatch, _pot_match))
                    vec = model[pot_match]
                    pair = [[nonmatch, vec]]
                    pair_arr = np.array(pair, dtype=object)
                    np.append(word_vec_pairs, pair_arr, axis=0)
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(non_matches, f)
                    break

            # Didn't find any similar words in the model
            # non_matches.remove(nonmatch)
            vec = None
            pair = [nonmatch, vec]
            pair_arr = np.array(pair, dtype=object)
            no_known_matches.append(pair_arr)
            with open(no_known_matches_pkl, 'wb') as f:
                pickle.dump(no_known_matches, f)

    print(non_matches)
    diff = abs(len(word_vecs) - len(words))
    print('Number of words missing from word2vec vocab: {}'.format(diff))
    with open('word_vectors.npy', 'wb') as f:
        np.save(f, word_vec_pairs)
    # df.save_csv('word_vecs.csv')

if __name__ == '__main__':
    main()
