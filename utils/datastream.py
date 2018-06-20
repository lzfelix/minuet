from collections import defaultdict
import numpy as np
import keras

class VariableBatchGenerator(keras.utils.Sequence):
    """Generator that grants that each batch contains sequences
    of the same length by relaxing the batch size.
    """
    
    def __init__(self, samples, labels, batch_size, min_batch_size,
                 shuffle=True, verbose=0):
        """Creates a generator object. Once this object is created,
        all batches are pre-computed, namely: no lazy evaluation is
        performed.
        
        :param samples: list of sequences, represented as lists.
        :param labels: list of sequence labels, represented as lists.
        :param batch_size: the suggested batch size. In reality the
        batch size is computed by the following formulas:
        
            amount_batches = int(np.ceil(amount_samples / self.batch_size))
            real_batch_size = int(np.ceil(amount_samples) / amount_batches)
            
        granting that the last batch is not too small.
        :param min_batch_size: if there are less sequences of a given
        size than this parameter, they are ignored.
        :param shuffle: if true samples are shuffled before each epoch.
        :param verbose: 0=silent, 1=show sequence length distribution,
        2=shows information about skipped samples.
        """
        
        self.X = samples
        self.y = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        
        self.verbose = verbose
        self.precomputed_batches = list()
        
        # organize sentences by length on buckets
        self.buckets = defaultdict(list)
        for index, sample in enumerate(self.X):
            self.buckets[len(sample)].append(index)

        if self.verbose == 1:
            sequence_frequencies = sorted(list(self.buckets.items()),
                                          key=lambda x:x[0])
            for seq_len, samples in sequence_frequencies:
                print('{}\t{}'.format(seq_len, len(samples)))
            
        # precomputing all batches
        self.__build_all_batches()
            
    def __build_all_batches(self):
        samples_counter = 0
        self.precomputed_batches = list()
        
        for samples_len, samples_indices in self.buckets.items():
            amount_samples = len(samples_indices)

            if amount_samples < self.min_batch_size:
                samples_counter += amount_samples
                if self.verbose == 2:
                    print(f'Skipping {amount_samples} samples '
                          'with size {samples_len}')
                continue

            # shuffling bucket as well
            if self.shuffle:
                np.random.shuffle(samples_indices)

            # batch_size is just a hint for the actual batch size, which
            # divides the samples into groups with the same amount of samples
            # avoiding the last batch to be too small.
            amount_batches = int(np.ceil(amount_samples / self.batch_size))
            real_batch_size = int(np.ceil(amount_samples) / amount_batches)

            if self.verbose == 2:
                print(f'{amount_samples} samples with length {samples_len}')
                print(f'{amount_batches} batches can be generated')
                print(f'{real_batch_size} samples per batch (hint was '
                      '{self.batch_size})')

            # here begins the generator part
            for batch_no in range(amount_batches):
                lower = batch_no * real_batch_size
                upper = min((batch_no + 1) * real_batch_size, amount_samples)

                X_batch = list()
                Y_batch = list()

                for index in samples_indices[lower:upper]:
                    X_batch.append(self.X[index])
                    Y_batch.append(self.y[index])

                X_batch = np.asarray(X_batch)
                Y_batch = np.asarray(Y_batch)

                self.precomputed_batches.append((X_batch, Y_batch))
                samples_counter += (upper - lower)
        
    def __len__(self):
        """Returns the amount of batches provided by this generator."""
        
        amount_ticks = 0
        for samples_indices in self.buckets.values():
            amount_samples = len(samples_indices)
            if amount_samples >= self.min_batch_size:
                amount_ticks += int(np.ceil(amount_samples / self.batch_size))
        return amount_ticks
    
    def __getitem__(self, i):
        return self.precomputed_batches[i]
    
    def on_epoch_end(self):
        """Shuffles the buckets between iterations."""
        
        if self.shuffle:
            np.random.shuffle(self.buckets)
