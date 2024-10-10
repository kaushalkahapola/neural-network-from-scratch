import numpy as np
import pandas as pd

from SoftmaxActivation import SoftmaxActivation
from CategoricalCrossEntropy import CategoricalCrossEntropy

class SoftmaxCategorical():
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalCrossEntropy()
    # Forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
