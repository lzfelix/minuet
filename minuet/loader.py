import re
import logging
import itertools
from typing import Tuple, List, Set, Dict

import numpy as np
import gensim


def load_dataset(filepath: str) -> Tuple[List[list], List[list]]:
    """Loads a file on the CoNLL dataset."""
    
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


def get_possible_labels(Y: List[List[str]]) -> List[str]:
    """Computes the set of unique labels from the dataset labels."""
    
    return list(set(itertools.chain(*Y)))


def get_characters_mapping(X, f=None):
    """Determines all unique characters from the dataset.
    
    :param X: List of tokenized sentences.
    :param f: Preprocessing function applied to every token before analysis.
    This function can be different from the one used with get_vocabulary.
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


def get_vocabulary(X: List[List[str]], f) -> Set[str]:
    """Determine unique words on the dataset.
    
    :param X: List of tokenized sentences.
    :param *preprocess: p_-like preprocessing functions.
    """
    
    f = f or (lambda x: x)
    
    vocab = set(['<unk>', '<num>'])
    for i, word in enumerate(itertools.chain(*X)):
        word_ = f(word)
        vocab.add(word_)

    return vocab


def load_embeddings(filepath: str, vocabulary: Set[str]) -> Tuple[Dict, np.ndarray]:
    """
    Loads the word embeddings for the necessary words only. Words not known by the
    model are skipped.
    
    :param filepath: Path to file with embedding vectors on gensim formet.
    :param vocabulary Vocabulary to be mapped to word vectors.
    :returns (V, E) where V maps words to their row number on the embedding matrix E.
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

    for word in vocabulary:
        if word in model:
            add_entry(word, model[word])

    vocabulary = vocabulary.intersection(word2index.keys())
    return word2index, np.asarray(word_vectors)
