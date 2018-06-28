import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

UNK_WORD_INDEX = 1


def sentence_to_characters(X, character_map, word_padding, sent_padding=0, f=None):
    """Computes the character index for each word in a sentence.
    Each word in a sentence is mapped to an array of character index. These
    index are grouped in another array, which described a single sentence
    of the dataset.
    
    :param X: A list of sentences, where each is a list of words already.
    :param character_map: A dict mapping characters to their IDs, the ID 0 is
    expected to correspond to the padding token <pad>.
    :param word_padding: The final length of each array of chars, smaller words
    are padded on the *left* with zeros. For larger arrays, only the first n
    elements on the *left* are kept.
    :param sent_padding: Every sentence in terms of chars representation will be
    adjusted to have this amount of words. If None or 0, the lenght of each sentence
    is left as is.
    :param f: A preprocessing function applied to each word before performing
    any computation. This preprocessing might be different from the one used
    to compute sentence representations on the word level.
    :param pad_sentence: If True 
    
    :returns If sent_padding is specified, returns a n-s-l matrix where n is the 
    amount of rows in X (samples), s is the sent_padding parameter and l is the
    word_padding parameter. Otherwise retuns a list of n elements, each a ?-l matrix
    where ? is the variable sentence length.
    """
    
    def sentence_longest_word(sentence, limit):
        """Returns sentence longest word and its length."""
        max_len = 0
        longest_word = ''
        for word in sentence:
            word_len = len(word)
            if max_len < word_len and word_len < limit:
                max_len = word_len
                longest_word = word
        return max_len, longest_word
    
    def corpus_longest_word(X, limit):
        """Returns the corpus longest word and its lenght."""
        # Finding corpus longest word
        max_len = 0
        longest_word = ''
        for sentence in X:
            sent_wordlen, sent_word = sentence_longest_word(sentence, limit)
            if max_len < sent_wordlen:
                max_len = sent_wordlen
                longest_word = sent_word

        return max_len, longest_word
    
    def adjust_encoded(encoded, maxlen, filler):
        if len(encoded) > maxlen:
            return encoded[:maxlen]
        return filler * (maxlen - len(encoded)) + encoded
    
    def adjust_encoded_word(encoded, maxlen):
        """Either pad or trim a word sequence of chars to have maxlen characters."""
        return adjust_encoded(encoded, maxlen, [0])
    
    dummy_word = [[0] * word_padding]
    def adjust_encoded_sent(encoded, maxlen):
        """Either pad or trim a sentence to have maxlen words."""
        return adjust_encoded(encoded, maxlen, dummy_word)

    f = f or (lambda x: x)
    max_len, longest_word = corpus_longest_word(X, word_padding)

    encoded_corpus = list()
    for sentence in X:
        encoded_sentence = list()
        for word in sentence:
            encoded_word = list()
            for char in f(word):
                encoded_word.append(character_map.get(char, 1))
                
            encoded_word = adjust_encoded_word(encoded_word, word_padding)
            encoded_sentence.append(encoded_word)
            
        if sent_padding:
            encoded_sentence = adjust_encoded_sent(encoded_sentence, sent_padding)
        encoded_corpus.append(np.asarray(encoded_sentence))

    if sent_padding:
        encoded_corpus = np.asarray(encoded_corpus)
    return encoded_corpus


def sentence_to_index(X, word2index, pre, max_sequence_len=None):
    """Converts sentences as lists of words to list of word ids.
    :param X: List of lists of words, each representing a sentence.
    :param word2index: Dict mapping each word to its ID.
    :param pre: Preprocessing functions to be applied to each word of X.
    :param max_sequence_len: Smaller sequences will be prepended with zeros and
    larger ones will be trimmed at the end. If None, no adjustment is performed.
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
            padding='pre')
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
