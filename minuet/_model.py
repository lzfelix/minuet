from keras import callbacks

from keras.layers import concatenate
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras_contrib.layers import CRF

class DeepModel():
    
    def build_char_embedding(self, amount_chars, char_embed_size, lstm_size, lstm_drop):
        """Builds the bottom part of the model responsible for character embedding.
        :param amount_chars: How many chars there are to embed.
        :param char_embed_size: The dimensionality of the char-embedded vector.
        :param lstm_size: size of the LSTM hidden state in which the char embedding
        vectors are fed to. This value is multiplied by 2 due to the BiLSTM.
        :param lstm_drop: dropout parameter for the BiLSTM.
        :return [feed, out]
        """
        
        
        # shape_in: (batch_size, sentence_words_maxlen, chars_maxlen [ids])
        chars_input = Input(shape=(None, None), name='char_input')
        
        char_embedding = Embedding(amount_chars, char_embed_size,
                                   embeddings_initializer='glorot_uniform',
                                   mask_zero=True, name='char_embedding'
        )(chars_input)
        #shape_out: (batch_size, sentence_maxlen, chars_maxlen, char_embedding_dim)
        
        char_embedding = TimeDistributed(
            Bidirectional(
                LSTM(lstm_size, dropout=lstm_drop,
                     recurrent_dropout=lstm_drop,
                     name='char_LSTM'
                )
            ),
            name='char_BiLSTM'
        )(char_embedding)
        # shape_out: (batch_size, sentence_words_maxlen, charlstm_hidden_dim_size)
        
        return chars_input, char_embedding
        
    def build_word_embedding(self, E):
        """Computes simple word embedding as a lookup layer for words in a sentence.
        :param E the embedding matrix with shape vocab_size-word_vec_dim.
        :return [feed, out]
        """
        
        
        # shape_in: (batch_size, sentence_words_maxlen)
        words_input = Input(shape=(None,), name='sent_input')
        
        word_embedding = Embedding(E.shape[0], E.shape[1],
                                   weights=[E], trainable=False,
                                   mask_zero=True, name='word_embedding'
        )(words_input)
        # shape_out: (batch_size, sentence_words_maxlen, word_embedding_dim)
        
        return words_input, word_embedding
    
    def build_sentence_lstm(self, word_embedding, char_embedding, lstm_size, drop_proba, bidirectional):
        """Computes the hidden vectors for each word in a sentence.
        :param word_embedding: The output of the word embedding part of the model.
        :param char_embedding: The output of the char embedding part of the model (optional). If not
        informed the model will not use character embedding. If provided these vectors are concatenated
        to the word embeddings.
        :param lstm_size: Size of the LSTM hidden state used to compute the contextual sentence
        representation. Each timestep hidden state is returned by this graph sub-module.
        :param drop_proba: LSTM dropout probability.
        :param bidirectional: If true a BiLSTM is used to read the sentence, duplicating the lenght of
        each timestep vector.
        :return [feed, out]
        """
        
        if char_embedding is None:
            word_representations = word_embedding
        elif word_embedding is None:
            word_representations = char_embedding
        else:
            word_representations = concatenate([word_embedding, char_embedding], axis=-1)

        # d = (word_embedding_dim + char_embedding_dim)
        # shape_in: (batch_size, sentence_words_maxlen, d)
        lstm = LSTM(lstm_size, dropout=drop_proba,
                    recurrent_dropout=drop_proba,
                    return_sequences=True, name='sent_LSTM'
        )
        
        if bidirectional:
            lstm = Bidirectional(lstm, name='sent_BiLSTM')
        sentence_representations = lstm(word_representations)
        # shape_out: (batch_size, sentence_words_maxlen, d*[1 or 2])
        
        return sentence_representations
    
    def build_softmax_output(self, sentence_representations, n_labels):
        """Builds the model's output layer formed by a softmax. Predictions are independent.
        :param sentence_representation: The output of the module <build_sentence_lstm>.
        :param n_labels: The amount of possible classes for the problem
        :return [feed, out]
        """
        
        # shape_in: (batch_size, sentence_words_maxlen, lstm_hidden_size*(1 or 2))
        # By default Dense only operates on the last layer
        out = Dense(n_labels, activation='softmax',
                    name='softmax')(sentence_representations)

        return out, 'sparse_categorical_crossentropy', ['sparse_categorical_accuracy']
        # shape_out: (batch_size, sentence_words_maxlen, n_classes)
        
    def build_crf_output(self, sentence_representations, n_labels):
        """Builds the model's output layer formed by a CRF. The last prediction affects the current.
        :param sentence_representation: The output of the module <build_sentence_lstm>.
        :param n_labels: The amount of possible classes for the problem
        :return [feed, out]
        """
        
        crf = CRF(n_labels, sparse_target=True, name='CRF')
        out = crf(sentence_representations)

        return out, crf.loss_function, [crf.accuracy]

    def create_callbacks(self, patience_delta, patience_wait, model_filepath):
        """Creates patience and checkpoint callbacks to train the model."""
        
        patience = callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=patience_delta,
                                           patience=patience_wait)
        all_callbacks = [patience]
        
        if model_filepath:
            checkpoint = callbacks.ModelCheckpoint(
                model_filepath, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=False, mode='auto'
            )
            all_callbacks.append(checkpoint)
            
        return all_callbacks
