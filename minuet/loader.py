import re
import logging
import itertools
from typing import Tuple, List, Set, Dict

import numpy as np
import gensim


def load_dataset(filepath):
    """Loads a file on the CoNLL dataset.

    :param filepath: Path to a text file on the CoNLL format.
    :return (X, Y) lists of sentences and labels.
    """
    
    X = list()
    x = list()

    Y = list()
    y = list()
    
    for line in open(filepath):
        # blank lines separate sequences
        if len(line) <= 1:
            X.append(x)
            Y.append(y)

            x = list()
            y = list()
        else:
            a, b = line.strip().split('\t')
            x.append(a)
            y.append(b)
    
    return X, Y


def get_possible_labels(Y):
    """Computes the set of unique labels from the dataset labels."""
    
    return list(set(itertools.chain(*Y)))


def get_characters_mapping(X, f=None):
    """Determines all unique characters from the dataset.
    
    :param X: List of tokenized sentences.
    :param f: Preprocessing functions applied to every token before analysis.
    This function can be different from the one used with get_vocabulary
    :return dict mapping characters to their IDs. Character 0 is always padding
    and 1 is always UNK
    """
    f = f or (lambda x: x)
    
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
    }
    for sentence in X:
        for word in sentence:
            for letter in f(word):
                if letter not in vocab:
                    vocab[letter] = len(vocab)
    return vocab


def get_vocabulary(X, f):
    """Determine unique words on the dataset.
    
    :param X: List of tokenized sentences
    :param *preprocess: p_-like preprocessing functions
    """
    
    f = f or (lambda x: x)
    
    vocab = set(['<unk>', '<num>'])
    for i, word in enumerate(itertools.chain(*X)):
        word_ = f(word)
        vocab.add(word_)

    return vocab


def load_embeddings(filepath, vocabulary, retain):
    """
    Loads the word embeddings for the necessary words only. Words not known by
    the model are skipped.
    
    :param filepath: Path to file with embedding vectors on gensim formet
    :param vocabulary: Vocabulary to be mapped to word vectors
    :param retain: If True, Minuet will keep all embeddings, otherwise it will
    keep just the vectors seen on training data
    :return (V, E) where V maps words to their row on the embedding matrix E
    """
    
    word2index = dict()
    word_vectors = list()

    def add_entry(word, vector):
        word2index[word] = len(word2index)
        word_vectors.append(vector)

    model = gensim.models.KeyedVectors.load(filepath)

    # adding special tokens <FIL>, <UNK> and <NUM>
    dim = model.vector_size
    add_entry('<fil>', np.zeros((dim,)))
    for special in ['<unk>', '<num>']:
        vector = np.random.uniform(-0.025, 0.025, (dim,))
        add_entry(special, vector)

    if retain:
        for word, _ in model.vocab.items():
            add_entry(word, model[word])
    else:
        for word in vocabulary:
            if word in model:
                add_entry(word, model[word])

    vocabulary = vocabulary.intersection(word2index.keys())
    return word2index, np.asarray(word_vectors)
