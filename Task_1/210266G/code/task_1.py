import numpy as np
import pandas as pd
from DenseLayer import DenseLayer
from ReluActivation import ReluActivation
from SoftmaxCategorical import SoftmaxCategorical
from CSVHandler import CSVHandler
import os


X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
label = np.array([3])

weights_path = os.path.join(os.path.dirname(__file__), '../../b/w-100-40-4.csv')
biases_path = os.path.join(os.path.dirname(__file__), '../../b/b-100-40-4.csv')

dweights_path = os.path.join(os.path.dirname(__file__), '../Task_1/dw.csv')
dbiases_path = os.path.join(os.path.dirname(__file__), '../Task_1/db.csv')

weights = pd.read_csv(weights_path, header=None)
weights = weights.T
weights.columns = weights.iloc[0]
weights = weights[1:]

biases = pd.read_csv(biases_path, header=None)
biases = biases.T
biases.columns = biases.iloc[0]
biases = biases[1:]

b1 = np.array(biases.iloc[:,0], dtype=np.float32)
b2 = np.array(biases.iloc[:40,1], dtype=np.float32)
b3 = np.array(biases.iloc[:4,2], dtype=np.float32)

w1 = np.array(weights.iloc[:,0:14], dtype=np.float32).T
w2 = np.array(weights.iloc[:40,14:114], dtype=np.float32).T
w3 = np.array(weights.iloc[:4,114:], dtype=np.float32).T

activation1 = ReluActivation()
dense1 = DenseLayer(14, 100, activation=activation1, weights=w1, biases=b1)
activation2 = ReluActivation()
dense2 = DenseLayer(100, 40, activation=activation2 ,weights=w2, biases=b2)
dense3 = DenseLayer(40, 4, weights=w3, biases=b3)
loss_activation = SoftmaxCategorical()

dense1.forward(X)
# activation1.forward(dense1.output)
dense1.activation.forward(dense1.output)
dense2.forward(dense1.activation.output)
# activation2.forward(dense2.output)
dense2.activation.forward(dense2.output)
dense3.forward(dense2.activation.output)
loss = loss_activation.forward(dense3.output, label)

loss_activation.backward(loss_activation.output, label)
dense3.backward(loss_activation.dinputs)
dense2.activation.backward(dense3.dinputs)
dense2.backward(dense2.activation.dinputs)
dense1.activation.backward(dense2.dinputs)
dense1.backward(dense1.activation.dinputs)

# Save gradients to CSV files
CSVHandler.write_to_csv(dbiases_path, dense1.dbiases.tolist(), dense2.dbiases.tolist(), dense3.dbiases.tolist())
CSVHandler.write_to_csv(dweights_path, dense1.dweights.tolist(), dense2.dweights.tolist(), dense3.dweights.tolist())

print("Gradients saved to CSV files")

