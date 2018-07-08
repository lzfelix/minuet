import json
from os import path

import numpy as np
from keras import models
from keras import optimizers
from keras_contrib.layers import CRF

from minuet._model import DeepModel
from minuet import encoder

class CharEmbeddingConfigs:
    
    def __init__(self, amount_chars, embedding_size, lstm_size, lstm_drop):
        self.amount_chars = amount_chars
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        
    def to_dict(self):
        return {
            'amount_chars': self.amount_chars,
            'char_embedding_size': self.embedding_size,
            'char_lstm_size': self.lstm_size,
            'char_lstm_drop': self.lstm_drop
        }


class Minuet():
    
    def __init__(self, embedding, lstm_size, lstm_drop, bidirectional=False, 
                 crf=False, char_embeddings_conf=None):
        """Creates a Bi-LSTM prediction model
        :param embedding: A v-by-d matrix where v is the vocabulary size and d
        the word-vectors dimension.
        :param lstm_size: The size of LSTM layer hidden vectors.
        :param lstm_drop: The variational LSTM dropout probability.
        :param bidirectional: Should the LSTM layer be bidirectional?
        :param char_embeddings_conf: If supplied, the model will use character
        embeddings as well.
        """
        
        self.E = embedding
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.bidirectional = bidirectional
        self.crf = crf
        self.n_labels = None
        
        self.char_embed = char_embeddings_conf
        
        self.model = None
        self._model_filepath = None
        self._model_folder = None
        
        self.deep = DeepModel()
        
        self.hyperparams = {
            'batch_size': 32,
            'epochs': 5
        }
        
    def _save_model_description(self, folder):
        """Serializes the object parameters as a lightweight JSON file.
        
        :param folder: The Circlet folder.
        :returns None
        """
        
        if not self.n_labels:
            raise RuntimeError('Amount of labels not defined yet.')
            
        if not folder:
            return
        
        description = {
            'word_vector_size': self.E.shape[1],
            'lstm_size': self.lstm_size,
            'lstm_dropout': self.lstm_drop,
            
            'bidirectional': self.bidirectional,
            'crf': self.crf,
            'amount_classes': self.n_labels,
        }
        description['char_embedding'] = vars(self.char_embed or {})
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
        
        # load model parameters stored on the Python object
        with open(path.join(model_folder, 'model.json')) as file:
            specs = json.load(file)

        lstm_size = specs['lstm_size']
        lstm_drop = specs['lstm_dropout']
        bidirectional = specs['bidirectional']
        crf = specs['crf']
        n_labels = specs['amount_classes']
        
        # load parameters specific to char embeddings
        char_specs = specs.get('char_embedding', None)
        if char_specs:
            char_configs = CharEmbeddingConfigs(
                char_specs['amount_chars'],
                char_specs['embedding_size'],
                char_specs['lstm_size'],
                char_specs['lstm_drop']
            )
        
        # creating a Minuet instance
        minuet = cls(None, lstm_size, lstm_drop, bidirectional, crf, char_specs)
        minuet.n_labels = n_labels
        minuet.model = model
        minuet._model_folder = model_folder
        minuet._model_filepath = model_filepath
        
        return minuet
        
    def set_checkpoint_path(self, model_folder):
        """Sets where the best Circlet model will be saved.
        :param model_folder: Path to a *folder* that will hold Circlet files.
        :returns None
        """
        
        self._model_folder = model_folder
        self._model_filepath = path.join(model_folder, 'model.hdf5')
        
    def _build_model(self):
        """Buils the model defined on the class initialization."""
        
        if self.model:
            return
        
        words_input, word_embedding = self.deep.build_word_embedding(self.E)
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

        self._save_model_description(self._model_folder)

        model_callbacks = self.deep.create_callbacks(1e-2, 3, self._model_filepath)
        self.history = self.model.fit(
            X, Y, validation_data=(X_val, Y_val),
            batch_size=self.hyperparams['batch_size'], epochs=self.hyperparams['epochs'],
            callbacks=model_callbacks
        )
        
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

    def predict(self, X, word2index, char2index, pre_word, pre_char, word_len):
        """Helper method to perform predictions.
        :param X: A list of samples, where each is a tokenized list of words.
        :param word2index: a dictionary mapping words to their row on the embedding matrix
        :param char2index: a dictionary mapping chars to their row on the embedding matrix
        :param pre_word: preprocessing pipeline for each sentence.
        :param pre_char: preprocessing pipeline applied to the characters of each word
        :param word_len: the maximum lenght of each word, if smaller it's padded, if larger
        it's trimmed.
        :return matrix n-by-l where n is the number of sentences in X and l is the size of the
        longest sentence in X, some of these labels are predictions over paddings and should be
        disregarded.
        """
        
        # make every sentence the size of the longest one
        sent_len = max(len(x) for x in X)

        # encode sentences and words (if used)
        sample_words = encoder.sentence_to_index(X, word2index, pre_word, sent_len)
        if word_len:
            sample_chars = encoder.sentence_to_characters(X, char2index, word_len, sent_len, pre_char)
            inputs = [sample_words, sample_chars]
        else:
            inputs = [sample_words]

        predictions = self.model.predict(inputs)
        return np.argmax(predictions, axis=-1)
