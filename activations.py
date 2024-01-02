import numpy as np

from layers import Activation

class Sigmoid(Activation):
    ''' Sigmoid activation class '''
    
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        activation = self(x)
        
        return activation * (1 - activation)
    
class TanH(Activation):
    ''' TanH activation class '''
      
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1
    
    def gradient(self, x):
        return 1 - self(x) ** 2
    
class ReLU(Activation):
    ''' ReLU activation class '''
    
    def __call__(self, x):
        return np.where(x >= 0, x, 0)
    
    def gradient(self, x):
        return np.where(x >= 0, 1, 0)
    
class LeakyReLU(Activation):
    ''' LeakyReLU activation class '''
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)
    
    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
    
class Softmax(Activation):
    ''' Softmax activation class '''
    
    def __call__(self, x):
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x / np.sum(x, axis=-1, keepdims=True)
    
    def gradient(self, x):
        activation = self(x)
        return activation * (1 - activation)
    