

import numpy as np
from activations import ReLU, Sigmoid, Tanh


class MLP:
    
    
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, 
                 activation='relu', seed=42):
        
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_type = activation
        
       
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(self.layer_dims) - 1
        
       
        self.activation = self._get_activation(activation)
        
        
        self.weights = {}
        self.biases = {}
        self._initialize_parameters()
        
      
        self.cache = {}
        
      
        self.gradients = {}
    
    def _get_activation(self, activation_type):
       
        if activation_type.lower() == 'relu':
            return ReLU()
        elif activation_type.lower() == 'sigmoid':
            return Sigmoid()
        elif activation_type.lower() == 'tanh':
            return Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")
    
    def _initialize_parameters(self):
       
        for i in range(self.num_layers):
            layer_idx = i + 1
            input_size = self.layer_dims[i]
            output_size = self.layer_dims[i + 1]
            
           
            if self.activation_type.lower() == 'relu':
               
                std = np.sqrt(2.0 / input_size)
            else:
               
                std = np.sqrt(1.0 / input_size)
            
            self.weights[f'W{layer_idx}'] = np.random.randn(input_size, output_size) * std
            self.biases[f'b{layer_idx}'] = np.zeros((1, output_size))
    
    def forward(self, X, training=True):
       
        if training:
            self.cache = {}
            self.cache['A0'] = X  
        A = X
        
       
        for i in range(self.num_layers):
            layer_idx = i + 1
            A_prev = A
            
           
            Z = np.dot(A_prev, self.weights[f'W{layer_idx}']) + self.biases[f'b{layer_idx}']
            
           
            if layer_idx < self.num_layers:
                A = self.activation.forward(Z)
            else:
                A = Z  
            
           
            if training:
                self.cache[f'Z{layer_idx}'] = Z
                self.cache[f'A{layer_idx}'] = A
        
        return A
    
    def backward(self, dL_dout):
       
        batch_size = dL_dout.shape[0]
        self.gradients = {}
        
        dA = dL_dout
        
      
        for i in range(self.num_layers, 0, -1):
            layer_idx = i
            
           
            Z = self.cache[f'Z{layer_idx}']
            A_prev = self.cache[f'A{layer_idx - 1}']
          
            if layer_idx < self.num_layers:
                dZ = dA * self.activation.backward(Z)
            else:
          
                dZ = dA
            
           
            self.gradients[f'W{layer_idx}'] = np.dot(A_prev.T, dZ) / batch_size
            
          
            self.gradients[f'b{layer_idx}'] = np.sum(dZ, axis=0, keepdims=True) / batch_size
            
           
            if layer_idx > 1:
                dA = np.dot(dZ, self.weights[f'W{layer_idx}'].T)
        
        return self.gradients
    
    def add_l2_gradients(self, weight_decay):
       
        for i in range(1, self.num_layers + 1):
            layer_idx = i
            self.gradients[f'W{layer_idx}'] += weight_decay * self.weights[f'W{layer_idx}']
    
    def get_parameters(self):
      
        params = {}
        params.update(self.weights)
        params.update(self.biases)
        return params
    
    def get_gradients(self):
       
        return self.gradients
    
    def set_parameters(self, params):
      
        for key, value in params.items():
            if key.startswith('W'):
                self.weights[key] = value
            elif key.startswith('b'):
                self.biases[key] = value
    
    def save_weights(self, filepath):
       
        params = self.get_parameters()
        params['layer_dims'] = self.layer_dims
        params['activation_type'] = self.activation_type
        np.savez(filepath, **params)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
       
        data = np.load(filepath, allow_pickle=True)
        
       
        saved_dims = data['layer_dims'].tolist()
        if saved_dims != self.layer_dims:
            print(f"Warning: Loaded layer dimensions {saved_dims} don't match current {self.layer_dims}")
        
       
        for i in range(1, self.num_layers + 1):
            layer_idx = i
            self.weights[f'W{layer_idx}'] = data[f'W{layer_idx}']
            self.biases[f'b{layer_idx}'] = data[f'b{layer_idx}']
        
        print(f"Model weights loaded from {filepath}")
    
    def predict(self, X):
      
        logits = self.forward(X, training=False)
        predictions = np.argmax(logits, axis=1)
        return predictions
    
    def predict_proba(self, X):
       
        logits = self.forward(X, training=False)
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities
    
    def __repr__(self):
       
        info = f"MLP(\n"
        info += f"  Architecture: {' -> '.join(map(str, self.layer_dims))}\n"
        info += f"  Activation: {self.activation_type}\n"
        total_params = sum(w.size for w in self.weights.values()) + sum(b.size for b in self.biases.values())
        info += f"  Total Parameters: {total_params:,}\n"
        info += f")"
        return info