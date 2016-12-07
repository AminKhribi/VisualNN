from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm, nonneg, unitnorm
from keras import optimizers, regularizers, layers, models




def create_model_gen(n_layers=1,
                     input_dim=None,
                     output_dim=12, act_output='softmax', 
                     K=[150], D=[0, 0], act='relu',
                     w_cons=None, l1=0, l2=0,
                     init='he_normal', optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics_nn=['accuracy']):

    """
    General function to create a Keras nn:
    Args: 
        - n_layers: number of layers
        - input_dim: int, n_ layer first layer
        - output_dim: 1 for regression, nb_classes for clf
        - act_output: activation function for output layer, softmax for clf and linear or else for regression
        - K: list of shape n_layers of nb of neurones
        - D: list of shape n_layers+1 specifiying dropout 
        - w_cons: constraint function to add to add to learning (weight capping). see keras constraint for list
        - l1, l2: regularization parameters
        - init: initialization of weights, he_normal recommanded
        - optimizer: adadelta or adam recommanded.
        - loss: see keras loss for list
        - metrics_nn: list of metrics to compute at each eopch, see metrics keras for list

    """

    model = models.Sequential()
    model.add(Dropout(D[0], input_shape=(input_dim, )))

    for i in range(n_layers):
        model.add(layers.Dense(K[i],
                            # activation=act,
                            init=init,
                            W_constraint=w_cons,
                            W_regularizer=regularizers.WeightRegularizer(l1=l1, l2=l2),
                            name="hidden1_clf{}".format(i)))
        
        model.add(layers.normalization.BatchNormalization())

        model.add(layers.advanced_activations.PReLU())
        # model.add(Activation(act))

        model.add(Dropout(D[i+1]))

    model.add(layers.Dense(output_dim, init='normal', activation=act_output))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics_nn)

    return model




def create_model_lstm(n_layers=1, 
                     timesteps=None, n_features=None,
                     output_dim=12, act_output='softmax',
                     K=[150], D=[0, 0], act='relu',
                     w_cons=None, l1=0, l2=0,
                     X_learn_lstm=None
                     init='he_normal', optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics_nn=['accuracy']):

    """
    Same but for LSTM network
    """

    model = Sequential()

    model.add(Dropout(D[0], batch_input_shape=(None, X_learn_lstm[0].shape[0], X_learn_lstm[0].shape[1])))
    model.add(layers.normalization.BatchNormalization())

    for i in range(n_layers):
        model.add(layers.recurrent.LSTM(K[i], activation='relu',
                                             W_regularizer=regularizers.WeightRegularizer(l1=0, l2=0),
                                             W_constraint=w_cons,
                                             stateful=True if n_layers > 1 else False
                                             # return_sequences=True
                                        ))

        model.add(layers.normalization.BatchNormalization())

        model.add(Dropout(D[i+1]))

    model.add(layers.Dense(output_dim, init='normal', activation=act_output))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics_nn)

    return model

