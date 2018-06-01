from keras.preprocessing import sequence

UNK_WORD_INDEX = 1


def sentence_to_index(X, E, word2index, pre, max_sequence_len):
    """Converts sentences as lists of words to list of word ids.
    :param X: List of lists of words, each representing a sentence.
    :param E: The word-embedding matrix.
    :param word2index: Dict mapping each word to its ID.
    :param pre: Preprocessing functions to be applied to each word of X.
    :param max_sequence_len: Smaller sequences will be trailled with 0 and while
    larger ones will be trimmed at the end.
    :returns A m-by-d matrix of word IDs (from word2index) where m is the amount
    of samples and d the word-vectors size.
    """
    
    all_sentences = list()
    for sentence in X:
        sentence_indices = list()

        for word in sentence:
            word = pre(word)
            index = word2index.get(word, UNK_WORD_INDEX)
            sentence_indices.append(index)
        all_sentences.append(sentence_indices)
        
    return sequence.pad_sequences(all_sentences, max_sequence_len, truncating='post', padding='post')
