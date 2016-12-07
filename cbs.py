import numpy as np

from sklearn import preprocessing

from keras import callbacks


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))

        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


class PlotGraph(callbacks.Callback):

    def __init__(self, n_neurones):
        self.n_neurones = n_neurones

    def on_epoch_end(self, epoch, logs={}):

        weights = []

        for layer in self.model.layers:
            if len(layer.get_weights()) and ('BatchNormalization' not in str(layer)) and ('activation' not in str(layer)):
                weights.append(layer.get_weights()[0].transpose())

        med_weights = []
        best = []
        for k in range(len(weights)-1):
            # med_weights.append(np.median(weights[k], axis=1))
            # med_weights.append(np.median(weights[k+1], axis=0))

            mw1 = np.median(weights[k], axis=1)
            mw2 = np.median(weights[k+1], axis=0)
            med_weights.append((mw1 + mw2) / 2)

        network = NeuralNetwork(max(self.n_neurones))
        network.add_layer(self.n_neurones[0], weights=weights[0])
        for k in range(len(weights)-1):
            med_w = med_weights[k]

            sc = preprocessing.MinMaxScaler()
            med_w = sc.fit_transform(med_w)

            med_w = np.where(med_w < 0, 0, med_w)
            med_w = np.where(med_w > 1, 1, med_w)

            network.add_layer(self.n_neurones[k+1], weights=weights[k+1], med_weights=med_w)

            best.append(sum(med_w < (np.mean(med_w) + 1.6 * np.std(med_w))))

        network.add_layer(self.n_neurones[-1])

        network.draw("epoch {}, recommendation: {}".format(epoch, best), (15, 15))