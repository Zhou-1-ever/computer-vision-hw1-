

import numpy as np


class ReLU:
   
    
    def forward(self, Z):
      
        return np.maximum(0, Z)
    
    def backward(self, Z):
       
        return (Z > 0).astype(float)
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid:
    
    
    def forward(self, Z):
      
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
    
    def backward(self, Z):
       
        A = self.forward(Z)
        return A * (1 - A)
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh:
   
    
    def forward(self, Z):
       
        return np.tanh(Z)
    
    def backward(self, Z):
      
        A = self.forward(Z)
        return 1 - A ** 2
    
    def __repr__(self):
        return "Tanh()"



if __name__ == "__main__":
  
    Z = np.array([[-2, -1, 0, 1, 2],
                  [-1, 0, 1, 2, 3]])
    
    print("Testing Activation Functions")
    print("=" * 50)
    print(f"Input Z:\n{Z}\n")
    
    # ReLU
    relu = ReLU()
    print(f"ReLU forward:\n{relu.forward(Z)}")
    print(f"ReLU backward:\n{relu.backward(Z)}\n")
    
    # Sigmoid
    sigmoid = Sigmoid()
    print(f"Sigmoid forward:\n{sigmoid.forward(Z)}")
    print(f"Sigmoid backward:\n{sigmoid.backward(Z)}\n")
    
    # Tanh
    tanh = Tanh()
    print(f"Tanh forward:\n{tanh.forward(Z)}")
    print(f"Tanh backward:\n{tanh.backward(Z)}\n")
