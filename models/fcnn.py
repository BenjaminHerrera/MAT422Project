# Imports
import torch.nn.functional as F
import torch.nn as nn
import torch


class FCNNModel(nn.Module):
    """Model definition for a fully connected neural network"""

    def __init__(
        self: "FCNNModel",
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_rate: int,
    ):
        """Constructor method for the fully connected model

        Args:
            self (FullyConnectedNN): Instance method identification
            input_size (int): Size of the input layer
            sizes (list[int]): Size of model's input, hidden layers, and
                output
            output_size (int): Size of the output layer
            dropout_rate (int): Rate of the dropout
        """

        # Pull attributes and methods from the parent class
        super(FCNNModel, self).__init__()

        # Augment hidden_sizes
        hidden_sizes = [input_size] + hidden_sizes + [output_size]

        # Build the layers for the model
        self.layers = nn.ModuleList()
        # self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                # self.norms.append(nn.LayerNorm(hidden_sizes[i + 1]))
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self: "FCNNModel", x: torch.Tensor) -> torch.Tensor:
        """Forward propogation function for the model

        Args:
            self (FullyConnectedNN): Instance method identification
            x (torch.Tensor): Input of the model

        Returns:
            torch.Tensor: Result of the forward pass
        """
        # Turn input into floats
        x = x.float()

        # Iterate through all of the layers except the last one
        for i in range(len(self.layers) - 1):
            # Intialially pass the inputs to the linear layers
            x = self.layers[i](x)

            # Apply batch normalization, ReLU, and dropout if applicable
            if i < len(self.layers) - 2:
                # x = self.norms[i](x) # TODO: Figure out a system to use this
                x = F.relu(x)
                x = self.dropouts[i](x)

            # If not, just pass to ReLU
            else:
                x = F.relu(x)

        # Return the result of the forward pass
        return self.layers[-1](x)
