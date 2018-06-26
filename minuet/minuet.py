import json
from os import path

import numpy as np
from keras import models
from keras import callbacks
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Bidirectional
from keras_contrib.layers import CRF

BATCH_SIZE = 16
EPOCHS = 10


class Minuet():
    
    def __init__(self, embedding, lstm_size, lstm_drop, bidirectional=False, 
                 crf=False):
        """Creates a Bi-LSTM prediction model
        :param embedding: A v-by-d matrix where v is the vocabulary size and d
            the word-vectors dimension.
        :param lstm_size: The size of LSTM layer hidden vectors.
        :param lstm_drop: The variational LSTM dropout probability.
        :param bidirectional: Should the LSTM layer be bidirectional?
        """
        
        self.E = embedding
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.bidirectional = bidirectional
        self.crf = crf
        self.n_labels = None
        self.model = None
        
        self._model_filepath = None
        self._model_folder = None
        
        self.hyperparams = {
            'batch_size': 16,
            'epochs': 10
        }
        
    def _save_model_description(self, folder):
        """Serializes the object parameters as a lightweight JSON file.
        
        :param folder: The Circlet folder.
        :returns None
        """
        
        if not self.n_labels:
            raise RuntimeError('Amount of labels not defined yet.')
        
        description = {
            'word_vector_size': self.E.shape[1],
            'lstm_size': self.lstm_size,
            'lstm_dropout': self.lstm_drop,
            'bidirectional': self.bidirectional,
            'amount_classes': self.n_labels
        }
        
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
        """Build the model: Embedding > (Bi)?LSTM > Softmax
        :returns None
        """
        if self.model:
            return
        
        sentence_in = Input(shape=(None,), name='input_layer')

        # word-embedding only
        embedding = Embedding(
            self.E.shape[0], self.E.shape[1],
            weights=[self.E],
            trainable=False, name='embedding',
            mask_zero=True
        )(sentence_in)

        # LSTM part
        lstm = LSTM(
            self.lstm_size, dropout=self.lstm_drop,
            recurrent_dropout=self.lstm_drop, return_sequences=True,
            name='LSTM'
        )
        
        if self.bidirectional:
            lstm = Bidirectional(lstm, name='BiLSTM')
        lstm = lstm(embedding)

        if self.crf:
            crf = CRF(self.n_labels, sparse_target=True)
            output = crf(lstm)
            
            loss_fun = crf.loss_function
            metrics = [crf.accuracy]
        else:
            # Output part
            output = Dense(self.n_labels, activation='softmax',
                           name='output')(lstm)
            loss_fun = 'sparse_categorical_crossentropy'
            metrics = ['sparse_categorical_accuracy']

        # Compiling the model
        model = models.Model(inputs=[sentence_in], outputs=[output])
        model.compile('adam', loss=loss_fun, metrics=metrics)
        
        print(model.optimizer)
        model.summary()
        self.model = model
        
    def __create_callbacks(self):
        # Adding assistence callbacks

        patience = callbacks.EarlyStopping(
            monitor='val_loss', min_delta=1e-2, patience=3
        )
        model_callbacks = [patience]
        
        if self._model_filepath:
            checkpoint = callbacks.ModelCheckpoint(
                self._model_filepath, monitor='val_loss',
                verbose=1, save_best_only=True, save_weights_only=False, mode='auto'
            )
            model_callbacks.append(checkpoint)
            
            self._save_model_description(self._model_folder)
            
        return model_callbacks
        
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
        
        model_callbacks = self.__create_callbacks()
        
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
        
        self.n_labels = n_labels
        self._build_model()
        
        model_callbacks = self.__create_callbacks()
        
        print('3541')
        self.history = self.model.fit_generator(
            gen_train, validation_data=gen_dev,
            epochs=self.hyperparams['epochs'], 
            callbacks=model_callbacks,
        )
