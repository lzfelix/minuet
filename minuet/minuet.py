import json
import warnings
from os import path

import numpy as np
import cloudpickle as pickle
from seqeval import metrics as seq_metrics

from keras import models
from keras import optimizers

from minuet._model import DeepModel
from minuet import _utils
from minuet import encoder


class CharEmbeddingConfigs:

    def __init__(self, char2index,
                 preprocessing,
                 maxlen=10,
                 embedding_size=32,
                 lstm_size=16,
                 lstm_drop=0.5,
                 noise_proba=0):
        """Character embedding hyperparameters.

        :param char2index: dict mapping each char to their ID on the
        embedding matrix
        :param preprocessing: A preprocessing pipeline from minuet.preprocessing
        :param maxlen: The maximum amount of characters that a word can have,
        smaller words are padded, larger are trimmed
        :param embedding_size: Number of columns on the char embedding matrix
        :param lstm_size: Size of the LSTM vectors on each direction
        :param lstm_drop: Dropout prob for forward and recurrent connections
        :param noise_proba: Proba in which chars are replaced by the UNK token
        """

        self.amount_chars = len(char2index)
        self.char2index = char2index
        self.pre = preprocessing
        self.maxlen = maxlen
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.lstm_drop = lstm_drop
        self.noise_proba = noise_proba
        
    def to_dict(self):
        """Returns a dictionary representation of this object."""
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
                 lstm_size=32,
                 lstm_drop=0.5,
                 sent_noise_proba=0,
                 bidirectional=False,
                 crf=False,
                 char_embeddings_conf=None):
        """Configures the creation of the Minuet model.

        :param char2index: dict mapping each word to their row on the
        embedding matrix
        :param pre_word: a preprocessing pipeline from minuet.preprocessing
        :para word_embedding: pretrained word embedding
        :param lstm_size: The size of LSTM layer hidden vectors
        :param lstm_drop: Dropout prob for forward and recurrent connections
        :param sent_noise_proba: Proba in which chars are replaced by UNK
        :param bidirectional: Should the LSTM layer be bidirectional?
        :param crf: Should the output be performed by CRF instead of softmax?
        :param char_embeddings_conf: If supplied, the model will use character
        embeddings as well
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

        self._label_encoder = None
        
        self.deep = DeepModel()
        
        self.hyperparams = {
            'batch_size': 32,
            'epochs': 5
        }

    def _save_model_description(self, folder):
        """Persists the model on disk.

        :param folder The model folder. It should already exist
        :return None
        """

        if not self.n_labels:
            raise RuntimeError('Amount of labels is not defined yet.')

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
        minuet_copy._label_encoder = self._label_encoder

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
        :return The restored Minuet model
        """
                
        # load model architecture and weights
        model_filepath = path.join(model_folder, 'model.hdf5')
        model = _utils.load_keras_model(model_filepath)

        # load Minuet object
        with open(path.join(model_folder, 'model.pyk'), 'rb') as file:
            minuet = pickle.load(file)
        minuet.model = model

        return minuet

    def set_label_encoder(self, label_encoder):
        """Stores the label encoder, making it being saved along the model.
        
        :param label_encoder: Trained instance of a SequenceLabelEncoder
        :return None
        """

        if not isinstance(label_encoder, encoder.SequenceLabelEncoder):
            raise RuntimeError('SequenceLabelEncoder expected.')
        self._label_encoder = label_encoder
        
    def set_checkpoint_path(self, model_folder):
        """Sets the *folder* (which should already exist) where the best Minuet
        model will be saved. 

        :param model_folder: Path to a *folder* that will hold Circlet files.
        :returns None
        """
        
        self._model_folder = model_folder
        self._model_filepath = path.join(model_folder, 'model.hdf5')
        
    def _build_model(self):
        """Buils the model defined during class initialization."""
        
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
            out, loss, acc = self.deep.build_crf_output(sentence_embeddings,
                                                        self.n_labels)
        else:
            out, loss, acc = self.deep.build_softmax_output(sentence_embeddings,
                                                            self.n_labels)

        self.model = models.Model(inputs=model_inputs, outputs=[out])
        self.model.compile('adam', loss=loss, metrics=acc)
        self.model.summary()
        
    def fit(self, X, Y, X_val, Y_val):
        """Fits the model. Notice that the index 0 for X should be reserved
        for padding sentences.

        :param X: If the model uses character embeddings, [W, C], otherwhise
        [W]. W is a matrix of sentence word embeddings, where each row is a
        sample and each column a word. C is a matrix of sentence character
        embeddings, where each row is a sample, each column a word and each
        volume the IDs of its characters. You can simply pipe in the output of
        the method prepare_samples
        :param Y A 3D a-b-c matrix, where a is the amount of samples, b the
            sequence size and c=1 (ie: amount of possible labels per sample)
        :param X_val Same as X, but for the validation set
        :param Y_val Same as Y, but for the validation set
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
        """Currently unavailable functionality."""
        
        raise NotImplementedError('Coming soon(TM)')
        
        # self.n_labels = n_labels
        # self._build_model()
        
        # model_callbacks = self.deep.create_callbacks(1e-2, 3, self._model_filepath)
        # self._save_model_description(self._model_folder)
        
        # self.history = self.model.fit_generator(
        #     gen_train, validation_data=gen_dev,
        #     epochs=self.hyperparams['epochs'], 
        #     callbacks=model_callbacks,
        # )

    def predict(self, X):
        """Predicts over a set of samples.

        :param X: If the model uses character embeddings, [W, C], otherwhise
        [W]. W is a matrix of sentence word embeddings, where each row is a
        sample and each column a word. C is a matrix of sentence character
        embeddings, where each row is a sample, each column a word and each
        volume the IDs of its characters. You can simply pipe in the output of
        the method prepare_samples

        :return n-by-l integer matrix, where n is the number of rows in X and l
        the length of the longest sample. Predictions might be padded with noise
        labels which should be removed
        """
        
        # make every sentence the size of the longest one
        sent_len = max(len(x) for x in X)

        inputs = self.prepare_samples(X, sent_len)
        y_hats = np.argmax(self.model.predict(inputs), axis=-1)
        x_lens = np.asarray([len(x) for x in X])

        # cropping the predictions matrix, so the sequence of labels
        # matches the sequence of words
        trimmed = list()
        for preds, lens in zip(y_hats, x_lens):
            trimmed.append(preds[:lens])

        return trimmed


    def prepare_samples(self, X, sent_maxlen):
        """Prepare samples to be classified by the model.

        :param X: List of sentences, where each sentence is a list of words
        :return W or [W, C]: where W are the word embeddings and C the char
        embeddings matrices (if applicable)
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

    def decode_predictions(self, predictions):
        """Converts class indices to string labels.
        
        :param predictions: Output of the method predict
        :return list of string sequence labels
        """

        self._supress_warnings()

        if not self._label_encoder:
            raise RuntimeError('Label decoder not found.')

        return self._label_encoder.inverse_transform(predictions)

    def evaluate(self, X, Y):
        self._supress_warnings()

        Y_hat = self.decode_predictions(self.predict(X))
        Y_hat = [y.tolist() for y in Y_hat]

        return seq_metrics.classification_report(Y, Y_hat) 

    def _supress_warnings(self):
        # We need to supress warnings. Can't downgrade numpy because of seqeval
        # Trick from https://stackoverflow.com/questions/49545947/sklearn-
        # deprecationwarning-truth-value-of-an-array
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

