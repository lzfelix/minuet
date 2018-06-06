import os
import json
import numpy as np
from keras import callbacks
from keras.models import Model
from keras.layers import Embedding, InputLayer, LSTM, Dense, Input, Bidirectional

BATCH_SIZE = 16
EPOCHS = 10


class Circlet():
    
    def __init__(self, embedding, lstm_size, lstm_drop, bidirectional=False):
        self.E = embedding
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.bidirectional = bidirectional
        self._was_model_built = False
        self._model_filepath = None
        
    @classmethod
    def load(cls, model_filepath):
        raise NotImplementedError()
        
    def set_checkpoint_path(self, model_filepath):
        self._model_filepath = model_filepath

    def _build_model(self):
        if self._was_model_built:
            return
        
        self._was_model_built = True
        
        sentence_in = Input(shape=(None,), name='input_layer')

        # word-embedding only
        embedding = Embedding(
            self.E.shape[0], self.E.shape[1], weights=[self.E],
            trainable=False, name='embedding'
        )(sentence_in)

        # LSTM part
        lstm = LSTM(
            self.lstm_size, dropout=self.lstm_drop,
            recurrent_dropout=self.lstm_drop, return_sequences=True,
            name='LSTM'
        )
        
        if self.bidirectional:
            lstm = Bidirectional(lstm)
        lstm = lstm(embedding)

        # Output part
        output = Dense(self.n_labels, activation='softmax', name='output')(lstm)

        # Compiling the model
        model = Model(inputs=[sentence_in], outputs=[output])
        model.compile(
            'nadam', loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        
        model.summary()
        self.model = model
        
    def fit(self, X, Y, X_val, Y_val):
        self.n_labels = np.unique(Y).size
        self._build_model()
        
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
        
        # then training
        self.history = self.model.fit(
            X, Y, BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val),
            callbacks=model_callbacks
        ) 
