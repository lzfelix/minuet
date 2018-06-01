from keras import callbacks
from keras.models import Model
from keras.layers import Embedding, InputLayer, LSTM, Dense, Input, Bidirectional

BATCH_SIZE = 16
EPOCHS = 10


class Circlet():
    
    def __init__(self, embedding, lstm_size, lstm_drop, n_labels, bidirectional=False):
        self.E = embedding
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.n_labels = n_labels
        self.bidirectional = bidirectional

    def build_model(self):
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
        
        # Bidirectional LSTM (optional) part
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
        # Adding assistence callbacks
        patience = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2,
                                           patience=3)
        
        self.history = self.model.fit(
            X, Y, BATCH_SIZE, epochs=EPOCHS,
            callbacks=[patience], validation_data=(X_val, Y_val)
        ) 
