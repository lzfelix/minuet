import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

UNK_WORD_INDEX = 1


def sentence_to_index(X, word2index, pre, max_sequence_len=None):
    """Converts sentences as lists of words to list of word ids.
    :param X: List of lists of words, each representing a sentence.
    :param word2index: Dict mapping each word to its ID.
    :param pre: Preprocessing functions to be applied to each word of X.
    :param max_sequence_len: Smaller sequences will be trailled with 0 and while
    larger ones will be trimmed at the end. If None no adjustment is performed.
    :returns Depends on max_sequence_len. If None, returns a list of lists
    otherwise a m-by-d matrix of word IDs (from word2index) where m is the
    amount of samples and d the word-vectors size.
    """
    
    all_sentences = list()
    for sentence in X:
        sentence_indices = list()

        for word in sentence:
            word = pre(word)
            index = word2index.get(word, UNK_WORD_INDEX)
            sentence_indices.append(index)
        all_sentences.append(sentence_indices)
        
    if max_sequence_len:
        return sequence.pad_sequences(
            all_sentences,
            max_sequence_len,
            truncating='post',
            padding='post')
    else:
        return all_sentences


def make_labels_sequential(sequences):
    """Transforms a sequences of labels into 3d matrix."""
    return [np.expand_dims(s, -1) for s in sequences]


class SequenceLabelEncoder(LabelEncoder):
    """Extends sklearn LabelEncoder for sequences of labels."""
    
    def __init__(self):
        super().__init__()

    def fit(self, sequence):
        unique_items = set()
        for sample in sequence:
            unique_items.update(sample)

        super().fit(list(unique_items))
        return self

    def transform(self, sequence):
        f = super().transform
        return list([f(sample) for sample in sequence])

    def fit_transform(self, sequence):
        self.fit(sequence)
        return self.transform(sequence)
    
    def inverse_transform(self, sequences):
        itransformed = list()
        for sample in sequences:
            itransformed.append(super().inverse_transform(sample))

        return np.asarray(itransformed)
        

def encode_labels(Y, max_sequence_len=None, encoder=None):
    """Maps sequences of text labels to categorical values. Optionally pad them.
    
    :param Y: list of list of not-encoded labels.
    :param max_sequence_len: Smaller sequences will be trailled with 0 and while
    larger ones will be trimmed at the end. If None no adjustment is performed.
    :param encoder: Label encoder to be used. If not provided, a new
    SequenceLabelEncoder is created and fitted.
    :returns (y_, e) where e is the encoded, either provided or created by this
    function and y depends on max_sequence len. If None, returns a list of lists,
    otherwise a m-by-d matrix of word IDs (from word2index).
    """

    if not encoder:
        encoder = SequenceLabelEncoder()
        y_ = encoder.fit_transform(Y)
    else:
        y_ = encoder.transform(Y)
        
    if max_sequence_len:
        y_ = sequence.pad_sequences(y_, max_sequence_len, truncating='post', padding='post')
    return y_, encoder
