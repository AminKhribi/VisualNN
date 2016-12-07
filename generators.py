import numpy as np


def batch_generator_lstm(X, y, batch_size, shuffle):
    """
    To use with fit_generator method, generates batches for LSTM learning (nb_samples, nb_timesteps, nb_features)
    """

    number_of_batches = np.ceil(len(X)/batch_size)
    counter = 0
    sample_index = np.arange(len(X))
    if shuffle:
        np.random.shuffle(sample_index)

    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = np.array(X)[batch_index]

        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            counter = 0


def batch_generatorp_lstm(X, batch_size, shuffle):
    """
    Generates batchs for testing
    """

    number_of_batches = np.ceil(len(X)/batch_size)
    counter = 0
    sample_index = np.arange(len(X))
    if shuffle:
        np.random.shuffle(sample_index)

    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = np.array(X)[batch_index]

        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0




def batch_generator(X, y, batch_size, shuffle):
    """
    Same as before but for regular type input (nb_samples, nb_features)
    """

    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):

    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0