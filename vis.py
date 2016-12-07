from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


vertical_distance_between_layers = 3000
horizontal_distance_between_neurons = 150
neuron_radius = 50



class Neuron():
    def __init__(self, x, y, br):
        self.x = x
        self.y = y
        self.br = br
        self.ref = np.linspace(0, 1, 3)

    def draw(self):
        if str(self.br) != 'None':
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True, color=self.ref * self.br)
        else:
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)

        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights, med_weights, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.med_weights = med_weights
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            mw = self.med_weights[iteration] if str(self.med_weights) != 'None' else None
            neuron = Neuron(x, self.y, mw)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, name):
        self.layers = []
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.name = name

    def add_layer(self, number_of_neurons, weights=None, med_weights=None):
        layer = Layer(self, number_of_neurons, weights, med_weights, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self, title, size):
        self.fig = pyplot.figure(figsize=size)

        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.title(title)
        self.fig.savefig("{}.png".format(self.name))
        pyplot.hold(True)


