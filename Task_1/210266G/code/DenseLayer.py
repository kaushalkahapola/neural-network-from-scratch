import numpy as np
import pandas as pd

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=None, weights=None, biases=None):
        self.weights = np.random.randn(n_inputs, n_neurons) if weights is None else weights
        self.biases = np.zeros((1, n_neurons)) if biases is None else biases
        self.activation = activation
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)