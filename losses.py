from numpy.typing import NDArray

import numpy as np

class Loss:
    ''' Base class for loss functions. '''
    
    def __call__(self, y: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        ''' Compute the loss function '''
        
        raise NotImplementedError()
    
    def gradient(self, y: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        ''' Compute the gradient of the loss function '''
        
        raise NotImplementedError()
    
    def acurracy(self, y: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        ''' Compute the acurracy of the model '''
        
        return 0
    
class MeanSquaredLoss(Loss):
    ''' Mean squared error loss function '''
    
    def __call__(self, y, p):
        return 0.5 * ((p - y) ** 2)
    
    def gradient(self, y, p):
        return p - y
        
class CrossEntropyLoss(Loss):
    ''' Binary cross entropy loss function '''
    
    def __call__(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        return - y * np.log(p) - (1 - y) * np.log(1 - p)
    
    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        return - (y / p) + (1 - y) / (1 - p)
    
    def acurracy(self, y, p):
        y = np.argmax(y, axis=1)
        p = np.argmax(p, axis=1)
        
        return np.sum(y == p, axis=0) / len(y)
