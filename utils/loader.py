import os
import re
import logging
import itertools
from typing import Tuple, List, Set, Dict

import numpy as np


def load_dataset(filepath: str) -> Tuple[List[list], List[list]]:
    """Loads a file on the CoNLL dataset."""
    
    X = list()
    x = list()

    Y = list()
    y = list()
    
    for line in open(filepath):
        if len(line) <= 1:
            X.append(x)
            Y.append(y)

            x = list()
            y = list()
        else:
            a, b = line.split('\t')
            x.append(a)
            y.append(b)
    
    return X, Y


def get_possible_labels(Y: List[List[str]]) -> List[str]:
    """Computes the set of unique labels from the dataset labels."""
    
    return list(set(itertools.chain(*Y)))


def get_vocabulary(X: List[List[str]], lowercase=True, replace_numbers=True) -> Set[str]:
    
    # TODO: improve number parsing
    identity = lambda w: w
    _lower = lambda w: w.lower() if lowercase else identity
    _number = lambda w: re.sub('[0-9]+', '<num>', w) if replace_numbers else identity
    
    vocab = set(['<unk>', '<num>'])
    
    for i, word in enumerate(itertools.chain(*X)):
        word_ = _lower(_number(word))
        vocab.add(word_)

    return vocab


def load_embeddings(filepath: str, vocabulary: Set[str], dimension: int) -> Tuple[Dict, np.ndarray]:
    """Loads the word embeddings for the necessary words only.
    :param filepath: Path to the embedding file.
    :param vocabulary
    """
    
    word2index = dict()
    word_vectors = list()
    
    def add_entry(word, vector):
        word2index[word] = len(word2index)
        word_vectors.append(vector)
        
    # TODO: Add vectors for UNK, FILL and NUM
    word2index['<fil>'] = 0
    word_vectors.append(np.zeros((dimension,)))
    
    for special in ['<unk>', '<num>']:
        vector = np.random.uniform(-0.25, 0.25, (dimension,))
        add_entry(special, vector)
        
    # reading word vectors for common vocabulary words
    for i, line in enumerate(open(filepath, 'r')):
        entries = line.split(' ')
        word, vector = entries[0], entries[1:]
        
        if word in vocabulary:
            vector = np.asarray(vector)
            add_entry(word, vector)

    word_vectors = np.asarray(word_vectors, dtype=np.float32)
    
    return word2index, word_vectors
