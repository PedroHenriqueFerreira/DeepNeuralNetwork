from typing import Optional

from numpy.typing import NDArray
from numpy import float64

import numpy as np

class Optimizer:
    ''' Base class for optimizers. '''
    
    def __call__(self, params: list[NDArray[float64]], gradients: list[NDArray[float64]]) -> None:
        ''' Updates the parameters of the model '''
        
        raise NotImplementedError()
    
class SGDOptimizer(Optimizer):
    ''' Stochastic Gradient Descent Optimizer '''
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.m: Optional[list[NDArray[float64]]] = None
        
    def __call__(self, params, gradients):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        
        for i, (param, gradient) in enumerate(zip(params, gradients)):
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * gradient

            param -= self.learning_rate * self.m[i]
    
class AdamOptimizer(Optimizer):
    ''' Adaptive Moment Estimation Optimizer '''
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.eps = 1e-8
        
        self.m: Optional[list[NDArray[float64]]] = None
        self.v: Optional[list[NDArray[float64]]] = None
        
        self.t = 0
      
    def __call__(self, params, gradients):
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
            
        self.t += 1
        
        for i, (param, gradient) in enumerate(zip(params, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradient ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)