import json
from os import path

import numpy as np
from keras import models
from keras import optimizers

from minuet._model import DeepModel


class Minuet():
    
    def __init__(self, embedding, lstm_size, lstm_drop, bidirectional=False, 
                 crf=False, char_lstm_size=None, char_embed_size=None,
                 char_lstm_drop=0, char_vocab_size=None):
        """Creates a Bi-LSTM prediction model
        :param embedding: A v-by-d matrix where v is the vocabulary size and d
            the word-vectors dimension.
        :param lstm_size: The size of LSTM layer hidden vectors.
        :param lstm_drop: The variational LSTM dropout probability.
        :param bidirectional: Should the LSTM layer be bidirectional?
        """
        
        self.E = embedding
        self.bidirectional = bidirectional
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        
        self.char_embed_size = char_embed_size
        self.char_vocab_size = char_vocab_size
        self.char_lstm_size = char_lstm_size
        self.char_lstm_drop = char_lstm_drop
        
        self.crf = crf
        self.n_labels = None
        
        self.model = None
        self._model_filepath = None
        self._model_folder = None
        
        self.deep = DeepModel()
        
        self.hyperparams = {
            'batch_size': 16,
            'epochs': 5,
            'clipnorm': 1.0
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
            'amount_classes': self.n_labels,
        }
        description.update(self.hyperparams)
        
        with open(path.join(folder, 'model.json'), 'w') as file:
            json.dump(description, file, indent=4)
        
    @classmethod
    def load(cls, model_folder):
        """Loads a previously trained Circlet model from a folder.
        :param model_folder: Path to the folder containing the mode files.
        :returns A loaded circlet instance *without* the E field.
        """
        
        model_filepath = path.join(model_folder, 'model.hdf5')
        model = models.load_model(model_filepath)
        
        with open(path.join(model_folder, 'model.json')) as file:
            specs = json.load(file)
        lstm_size = specs['lstm_size']
        lstm_drop = specs['lstm_dropout']
        bidirectional = specs['bidirectional']
        n_labels = specs['amount_classes']
        
        # creating a Circlet instance
        circlet = cls(None, lstm_size, lstm_drop, bidirectional)
        circlet.n_labels = n_labels
        circlet.model = model
        circlet._model_folder = model_folder
        circlet._model_filepath = model_filepath
        
        return circlet
        
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
        chars_input, char_embedding = self.deep.build_char_embedding(
            self.char_vocab_size,
            self.char_embed_size,
            self.char_lstm_size,
            self.char_lstm_drop)
        
        sentence_embeddings = self.deep.build_sentence_lstm(word_embedding,
                                                            char_embedding,
                                                            self.lstm_size,
                                                            self.lstm_drop,
                                                            self.bidirectional)
        
        if self.crf:
            out, loss, acc = self.deep.build_crf_output(sentence_embeddings, self.n_labels)
        else:
            out, loss, acc = self.deep.build_softmax_output(sentence_embeddings, self.n_labels)
            
        opt = optimizers.Adam(clipnorm=self.hyperparams['clipnorm'])
        
        self.model = models.Model(inputs=[words_input, chars_input], outputs=[out])
        self.model.compile(opt, loss=loss, metrics=acc)
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
        self._save_model_description(self._model_folder)
        
        # then training
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
