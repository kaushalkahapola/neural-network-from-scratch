import numpy as np

class OptimizerSGD:
    def __init__(self, clip_value=None):
        self.clip_value = clip_value  # Add clipping option
    
    # Update parameters
    def update_params(self, layer, learning_rate):
        
        if self.clip_value is not None:
            np.clip(layer.dweights, -self.clip_value, self.clip_value, out=layer.dweights)
            np.clip(layer.dbiases, -self.clip_value, self.clip_value, out=layer.dbiases)

        layer.weights += -learning_rate * layer.dweights.astype(np.float32)
        layer.biases += (-learning_rate * layer.dbiases.astype(np.float32)).flatten()
