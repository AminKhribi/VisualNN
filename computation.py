from keras import callbacks, 

from scipy import sparse

from sklearn import cross_validation

from cbs import *
from generators import *

import matplotlib.pyplot as plt


def compute_nn(create_model=None,
			   batch_generator=None, batch_generatorp=None,
			   param_model=None,
			   param_fit=None,
			   X_learn=None, Y_learn=None,
			   X_test=None, Y_test=None,
			   score_func=None,
			   plot=True,
			   plot_graph=False, n_neurones=None, name_plot=None):

	"""
	Function to fit an NN to be constructed with create_model function.
	Learning is done with saving best model at each epoch and earlystopping when the error does not decrease after some epochs
	This can also plot accuracy and loss (if accuracy is specified in metrics)
	Finally, it returns fitted model and score with score_func
	Args:
		- create_model: instance of function of models_gen
		- batch_generator, batch_generatorp: instance of function in generators
		- param_model: dict of parameters to pass to create_model
		- param_fit: dict of learning params, kets are: batch_size, nb_epoch, validation_split, patience (nb_epochs to wait
		  before stopping learning)
		- X_learn, Y_learn, X_test, Y_test1
		- score_func: any function that returns a float and takes as input y_true and y_hat
		- plot: boolean to plot accuracy and loss by epoch
		- plot_graph: boolean to plot the architecture of the NN
		- n_neurones: input needed for the plot_graph, list of number of neurones by layer
	"""

    if not ('sparse' in str(type(X_learn))):
        X_learn = sparse.coo_matrix(X_learn).tocsr()
        X_test = sparse.coo_matrix(X_test).tocsr()

    # skf = list(cross_validation.StratifiedKFold(Y_all[idx_learn], 2))

    # s = 0
    # for i, (train, test) in enumerate(skf):

    #     X_learn1, X_test1 = X_learn[train], X_learn[test]
    #     Y_learn1, Y_test1 = Y_learn[train], Y_learn[test]

    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_learn, Y_learn, train_size=param_fit['validation_split'], random_state=10)

    model = create_model(**param_model)

    filepath = "weights{}.best.hdf5".format(random.random())
    cb1 = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    cb2 = callbacks.EarlyStopping(monitor='val_loss', patience=param_fit['patience'], verbose=1, mode='auto')

    history = LossHistory()

    cbs = [cb1, cb2, history]

    if plot_graph:
        plot_nn = PlotGraph(n_neurones, name_plot)
        cbs.append(plot_nn)

    fit = model.fit_generator(generator=batch_generator(X_train, y_train, param_fit['batch_size'], True),
                              nb_epoch=param_fit['nb_epoch'],
                              samples_per_epoch=X_train.shape[0],
                              validation_data=(X_val.todense(), y_val),
                              callbacks=cbs,
                              verbose=2
                              )

    if plot:

        try:
            os.mkdir("nn_all/{}".format(str(param_model)))

        except:
            return 0


        fig = plt.figure()
        plt.plot(range(len(history.train_losses)), history.train_losses, label='train')
        plt.hold(True)
        plt.plot(range(len(history.val_losses)), history.val_losses, label='val')
        plt.legend()
        fig.savefig("nn_all/{}/loss.png".format(str(param_model)))

        try:
	        fig = plt.figure()
	        plt.plot(range(len(history.train_acc)), history.train_acc, label='train')
	        plt.hold(True)
	        plt.plot(range(len(history.val_acc)), history.val_acc, label='val')
	        plt.legend()
	        plt.title("acc train/val")
	        plt.savefig("nn_all/{}/acc.png".format(str(param_model)))

	    except:
	    	print("accuracy is not specified")
    #
    model = create_model(**param_model)

    model.load_weights(filepath)

    # evaluate the model
    y_hat = model.predict_generator(generator=batch_generatorp(X_test, param_fit['batch_size'], False), val_samples=X_test.shape[0])
    score = score_func(Y_test, y_hat)

    # s += score / 2

    return score, model
