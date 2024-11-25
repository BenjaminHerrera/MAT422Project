# Imports
import torch


class RNNModel(torch.nn.Module):
    """Model definition for a Recurrent Neural Network"""

    def __init__(
        self: "RNNModel",
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_rate: float,
    ):
        """Constructor method for the recurrent neural network model

        Args:
            self (RNNModel): Instance method identification
            input_size (int): Size of the input layer
            hidden_size (int): Size of the hidden layers
            output_size (int): Size of the output layer
            num_layers (int): Number of RNN layers
            dropout_rate (float): Rate of the dropout
        """
        super(RNNModel, self).__init__()

        # Define the RNN layer
        self.rnn = torch.nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        # Define output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self: "RNNModel", x: torch.Tensor) -> torch.Tensor:
        """Forward propagation function for the model

        Args:
            self (RNNModel): Instance method identification
            x (torch.Tensor): Input of the model

        Returns:
            torch.Tensor: Result of the forward pass
        """
        # If input is 2-D (sequence_length, input_size),
        #   unsqueeze to add batch dimension
        # Shape becomes (1, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Initialize hidden state with zeros:
        #   (batch_size, num_layers, hidden_size)
        h0 = torch.zeros(
            self.rnn.num_layers, x.size(0), self.rnn.hidden_size
        ).to(x.device)

        # Get RNN outputs
        out, _ = self.rnn(x, h0)

        # Pass the RNN output (last time step) to the fully connected layer
        out = self.fc(out[:, -1, :])

        # Return the model's output
        return out
