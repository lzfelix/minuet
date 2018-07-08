import json
from os import path

import numpy as np
import cloudpickle as pickle
from keras import models
from keras import optimizers
from keras_contrib.layers import CRF

from minuet._model import DeepModel
from minuet import encoder

class CharEmbeddingConfigs:
    
    def __init__(self, char2index,
                 preprocessing,
                 maxlen,
                 embedding_size,
                 lstm_size,
                 lstm_drop,
                 noise_proba=0):
        """Character embedding hyperparameters."""

        self.amount_chars = len(char2index)
        self.char2index = char2index
        self.pre = preprocessing
        self.maxlen = maxlen
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.noise_proba = noise_proba
        
    def to_dict(self):
        return {
            'amount_chars': self.amount_chars,
            'embedding_size': self.embedding_size,
            'lstm_size': self.lstm_size,
            'lstm_drop': self.lstm_drop,
            'maxlen': self.maxlen,
            'noise_proba': self.noise_proba
        }


class Minuet():
    
    def __init__(self, word2index,
                 pre_word,
                 word_embedding,
                 lstm_size,
                 lstm_drop,
                 sent_noise_proba=0,
                 bidirectional=False,
                 crf=False,
                 char_embeddings_conf=None):
        """Creates a Bi-LSTM prediction model.
        :param embedding: A v-by-d matrix where v is the vocabulary size and d
        the word-vectors dimension.
        :param lstm_size: The size of LSTM layer hidden vectors.
        :param lstm_drop: The variational LSTM dropout probability.
        :param bidirectional: Should the LSTM layer be bidirectional?
        :param char_embeddings_conf: If supplied, the model will use character
        embeddings as well.
        """
        
        self._E = word_embedding
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.bidirectional = bidirectional
        self.crf = crf
        self.n_labels = None
        
        self.char_embed = char_embeddings_conf

        self.word2index = word2index
        self.pre_word = pre_word
        self.sent_noise_proba = sent_noise_proba
        
        self.model = None
        self._model_filepath = None
        self._model_folder = None
        
        self.deep = DeepModel()
        
        self.hyperparams = {
            'batch_size': 32,
            'epochs': 5
        }

    def _save_model_description(self, folder):
        if not self.n_labels:
            raise RuntimeError('Amount of labels is not defined yet.')
            
        if not folder:
            return

        # creating a Minuet copy without word embeddings
        minuet_copy = Minuet(
            self.word2index,
            self.pre_word,
            None,
            self.lstm_size,
            self.lstm_drop,
            self.sent_noise_proba,
            self.bidirectional,
            self.crf,
            self.char_embed
        )
        minuet_copy.n_labels = self.n_labels

        # pickling
        with open(path.join(folder, 'model.pyk'), 'wb') as file:
            pickle.dump(minuet_copy, file)

        # generating JSON description for manual inspection
        description = {
            'word_vector_size': self._E.shape[1],
            'lstm_size': self.lstm_size,
            'lstm_dropout': self.lstm_drop,
            'noise_proba': self.sent_noise_proba,
            
            'bidirectional': self.bidirectional,
            'crf': self.crf,
            'amount_classes': self.n_labels,
        }
        char_embed_data = self.char_embed.to_dict() if self.char_embed else dict()
        description['char_embedding'] = char_embed_data
        description['hyperparams'] = self.hyperparams

        with open(path.join(folder, 'model.json'), 'w') as file:
            json.dump(description, file, indent=4)

    @classmethod
    def load(cls, model_folder):
        """Loads a previously trained Minuet model from a folder.
        :param model_folder: Path to the folder containing the model files.
        :returns A loaded circlet instance *without* the E field.
        """
        
        # solution from https://github.com/keras-team/keras-contrib/issues/129
        def create_custom_objects():
            instanceHolder = {"instance": None}
            
            class ClassWrapper(CRF):
                def __init__(self, *args, **kwargs):
                    instanceHolder["instance"] = self
                    super(ClassWrapper, self).__init__(*args, **kwargs)
                    
            def loss(*args):
                method = getattr(instanceHolder["instance"], "loss_function")
                return method(*args)
            
            def accuracy(*args):
                method = getattr(instanceHolder["instance"], "accuracy")
                return method(*args)
            
            return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

        def load_keras_model(path):
            model = models.load_model(path, custom_objects=create_custom_objects())
            return model
        
        # load model architecture and weights
        model_filepath = path.join(model_folder, 'model.hdf5')
        model = load_keras_model(model_filepath)

        # load Minuet object
        with open(path.join(model_folder, 'model.pyk'), 'rb') as file:
            minuet = pickle.load(file)
        minuet.model = model

        return minuet
        
    def set_checkpoint_path(self, model_folder):
        """Sets where the best Circlet model will be saved.
        :param model_folder: Path to a *folder* that will hold Circlet files.
        :param word2index: Dicionarty mapping words to their IDs
        :returns None
        """
        
        self._model_folder = model_folder
        self._model_filepath = path.join(model_folder, 'model.hdf5')
        
    def _build_model(self):
        """Buils the model defined on the class initialization."""
        
        if self.model:
            return
        
        words_input, word_embedding = self.deep.build_word_embedding(self._E)
        model_inputs = [words_input]
        char_embedding = None
        
        if self.char_embed:
            chars_input, char_embedding = self.deep.build_char_embedding(
                self.char_embed.amount_chars,
                self.char_embed.embedding_size,
                self.char_embed.lstm_size,
                self.char_embed.lstm_drop)
            model_inputs = [words_input, chars_input]
        
        sentence_embeddings = self.deep.build_sentence_lstm(word_embedding,
                                                            char_embedding,
                                                            self.lstm_size,
                                                            self.lstm_drop,
                                                            self.bidirectional)
        
        if self.crf:
            out, loss, acc = self.deep.build_crf_output(sentence_embeddings, self.n_labels)
        else:
            out, loss, acc = self.deep.build_softmax_output(sentence_embeddings, self.n_labels)
            
        #opt = optimizers.Adam(clipnorm=self.hyperparams['clipnorm'])
        
        self.model = models.Model(inputs=model_inputs, outputs=[out])
        self.model.compile('adam', loss=loss, metrics=acc)
        self.model.summary()
        
    def fit(self, X, Y, X_val, Y_val):
        """Fits the model. Notice that the index 0 for X should be reserved
        for padding sentences.
        :param X An integer matrix where each row correspons to a sentence.
        :param Y A 3D a-b-c matrix, where a is the amount of samples, b the
            sequence size and c=1 (ie: amount of possible labels per sample)
        :param X_val The validation samples in the same shape as X.
        :param Y_val The validation labels in the same shape as Y.
        :returns None
        """

        self.n_labels = np.unique(Y).size
        self._build_model()

        model_callbacks = self.deep.create_callbacks(1e-2, 3, self._model_filepath)
        self.history = self.model.fit(
            X, Y, validation_data=(X_val, Y_val),
            batch_size=self.hyperparams['batch_size'], epochs=self.hyperparams['epochs'],
            callbacks=model_callbacks
        )
        self._save_model_description(self._model_folder)

    def fit_generator(self, gen_train, gen_dev, n_labels):
        """Fits the model using generators. Notice that the index 0 for X
        should be reserved for padding sentences.
        :param X An integer matrix where each row correspons to a sentence.
        :param Y A 3D a-b-c matrix, where a is the amount of samples, b the
            sequence size and c=1 (ie: amount of possible labels per sample)
        :param X_val The validation samples in the same shape as X.
        :param Y_val The validation labels in the same shape as Y.
        :returns None
        """
        
        raise NotImplementedError('Comming soon(TM)')
        
        self.n_labels = n_labels
        self._build_model()
        
        model_callbacks = self.deep.create_callbacks(1e-2, 3, self._model_filepath)
        self._save_model_description(self._model_folder)
        
        print('3541')
        self.history = self.model.fit_generator(
            gen_train, validation_data=gen_dev,
            epochs=self.hyperparams['epochs'], 
            callbacks=model_callbacks,
        )

    def predict(self, X):
        """Predicts over a set of samples.
        :param X: a list of sentences, where each sentence is a list of words.
        :return n-by-l matrix, where n is the number of sentences and l
        the length of the longest sample.
        """
        
        # make every sentence the size of the longest one
        sent_len = max(len(x) for x in X)

        inputs = self.prepare_samples(X, sent_len)
        return np.argmax(self.model.predict(inputs), axis=-1)

    def prepare_samples(self, X, sent_maxlen):
        """Prepare samples to be classified by the model.
        :param X: List of sentences, where each sentence is a list of words.
        :return u or [u, v]: where u are the word embeddings and v the char
        embeddings (if used).
        """
        X_words = encoder.sentence_to_index(X, self.word2index, self.pre_word, 
                                            sent_maxlen, self.sent_noise_proba)
        out = X_words
        if self.char_embed:
            ce = self.char_embed
            X_chars = encoder.sentence_to_characters(X, ce.char2index, ce.maxlen,
                                                     sent_maxlen, ce.pre, 
                                                     ce.noise_proba)
            out = [X_words, X_chars]
        return out 

