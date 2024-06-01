import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None, dropout_rate=0.3):
        """
        :param input_size: Size of the input layer
        :param output_size: Size of the output layer
        :param hidden_layers: A list where each element is the number of nodes in that hidden layer. 
        :param dropout_rate: The rate at which dropout will be applied.
        """
        super(MLP, self).__init__()
        
        # Default to one hidden layer with 50 nodes if not specified
        if hidden_layers is None:
            hidden_layers = [50]
        
        # Create the layers
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
        # Last hidden layer to output
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after activation
        x = self.layers[-1](x)
        return x
