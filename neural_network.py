from typing import Optional

from numpy.typing import NDArray
from numpy import float64

import numpy as np

from optimizers import Optimizer
from losses import Loss

from layers import Layer

class NeuralNetwork:
    ''' Neural Network class '''
    
    def __init__(self, optimizer: Optimizer, loss: Loss):
        self.layers: list[Layer] = []
        
        self.optimizer = optimizer
        self.loss = loss
        
    def add(self, layer: Layer) -> None:
        ''' Add layer to the network '''
        
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].output_shape)
            
        layer.initialize()
        
        self.layers.append(layer)
        
    def forward(self, input_value: NDArray[float64], training: bool = True) -> NDArray[float64]:
        ''' Get the output value from the network '''
        
        for layer in self.layers:
            input_value = layer.forward(input_value, training)
            
        return input_value
    
    def backward(self, output_gradient: NDArray[float64], training: bool = True) -> NDArray[float64]:
        ''' Get the input gradient from the network '''
        
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, training)
            
        return output_gradient
    
    def train_on_batch(self, X: NDArray[float64], y: NDArray[float64]) -> float64:
        ''' Train a batch of data and return loss '''
        
        y_pred = self.forward(X)
        self.backward(self.loss.gradient(y, y_pred))
        self.optimizer(self.parameters, self.gradients)
        
        loss: float64 = self.loss(y, y_pred).mean()
        
        return loss
    
    def not_train_on_batch(
        self, 
        X: NDArray[float64], 
        y: NDArray[float64]
    ) -> tuple[float64, NDArray[float64]]:
        ''' Not train a batch of data and return loss and input gradient '''
        
        y_pred = self.forward(X, training=False)
        input_gradient = self.backward(self.loss.gradient(y, y_pred), training=False)
        
        loss: float64 = self.loss(y, y_pred).mean()
        
        return loss, input_gradient
     
    def test_on_batch(self, X: NDArray[float64], y: NDArray[float64]) -> float64:
        ''' Return loss from a batch of data '''
        
        y_pred = self.forward(X, training=False)
        
        return self.loss(y, y_pred).mean() # type: ignore
    
    def fit(
        self, 
        X: NDArray[float64], 
        y: NDArray[float64], 
        epochs: int = 100, 
        batch_size: int = 64, 
        X_val: Optional[NDArray[float64]] = None,
        y_val: Optional[NDArray[float64]] = None,
        shuffle: bool = True
    ) -> None:
        ''' Train the network to fit the data '''
        
        size = X.shape[0]
        batchs = ((size - 1) // batch_size) + 1
        
        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1} / {epochs}\n')
            
            if shuffle:
                permutation = np.random.permutation(size)
                
                X = X[permutation]
                y = y[permutation]
            
            loss = []
            
            for batch in range(batchs):
                print(f'Batch: {batch + 1} / {batchs}')
                
                index = batch * batch_size
                
                X_batch = X[index : index + batch_size]
                y_batch = y[index : index + batch_size]
                
                loss.append(self.train_on_batch(X_batch, y_batch))
            
            print(f'Loss: {np.mean(loss)}')
            
            if X_val is not None and y_val is not None:
                val_loss = self.test_on_batch(X_val, y_val)
                
                print(f'Validation Loss: {val_loss}')
    
    def summary(self, name: str = 'Model', col_width: int = 25) -> None:
        ''' Print summary '''
        
        def print_row(layer_name, output_shape, params):
            layer_name = str(layer_name).ljust(col_width)
            output_shape = str(output_shape).ljust(col_width)
            params = str(params).ljust(col_width)
            
            print(f'{layer_name} {output_shape} {params}')
        
        def print_line(char: str = '-') -> None:
            print(char * 3 * col_width)
            
        print_line('=')
        
        print(f'{name} Summary')
        
        print_line()
        
        print_row('Layer Name', 'Output Shape', 'Parameters')
        
        print_line()
        
        input_shape = self.layers[0].input_shape
        
        print_row('Input Layer', (None, *input_shape), 0)
        
        total_params = 0
        
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            out_shape = (None, *layer.output_shape)
            params = layer.parameters()
            
            print_row(layer_name, out_shape, params)
            
            total_params += params
            
        print_line()
        
        print(f'Total Parameters: {total_params:,}')
        
        print_line('=')
    
    @property
    def parameters(self) -> list[NDArray[float64]]:
        ''' Get the parameters '''
        
        output: list[NDArray[float64]] = []
        
        for layer in self.layers:
            if layer.parameters() == 0:
                continue
            
            output.append(layer.weights) # type: ignore
            output.append(layer.bias) # type: ignore
            
        return output
    
    @property
    def gradients(self) -> list[NDArray[float64]]:
        ''' Get the gradients '''
        
        output: list[NDArray[float64]] = []
        
        for layer in self.layers:
            if layer.parameters() == 0:
                continue
            
            output.append(layer.weights_gradient) # type: ignore
            output.append(layer.bias_gradient) # type: ignore
            
        return output