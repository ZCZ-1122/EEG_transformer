"""Defines the neural network, losss function and metrics"""
from typing import Tuple

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


######### conv1d -> LSTM (Multiclass) ###########
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.input_channel = params.input_channel
        self.n_filters = params.n_filters
        self.kernel_size = params.kernel_size
        self.num_class = params.num_class

        self.lstm_hidden_size = params.lstm_hidden_size
        self.lstm_n_layers = params.lstm_n_layers

        self.dropout_p = params.dropout_p

        # conv_1d to reduce time steps
        self.conv_1d = nn.Conv1d(self.input_channel, self.n_filters, self.kernel_size, 
                                padding='valid', stride=2)

        # nn.Relu() ??
        
        # LSTM layers
        self.lstm = nn.LSTM(self.n_filters, self.lstm_hidden_size, self.lstm_n_layers, 
                            batch_first=True, dropout=self.dropout_p)
        
        # Linear layer for binary classification
        self.fc = nn.Linear(self.lstm_hidden_size, self.num_class)

    def forward(self, x):
    
        # x : shape (N, L, D) 

        # conv_1d requires input shape (N, C_in, L_in)
        # conv_1d returns output shape (N, C_out, L_out)
        out = self.conv_1d(x.transpose(2,1)).transpose(2,1)

        # LSTM(batch_first=True), expected input shape (N, L, D)
        out, _ = self.lstm(out) 
        
        # Index the output of the last time step
        out = out[:, -1, :] # batch_first=True, out has shape (N, L, H_out)
        
        # Apply the linear transformation for binary classification
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)

        # Get col idx range (i) and powers
        i = torch.arange(max_len)[:, None]
        pows = torch.pow(10000, -torch.arange(0, embed_dim, 2) / embed_dim)

        # Compute positional values sin/cos
        pe[0, :, 0::2] = torch.sin(i * pows)
        pe[0, :, 1::2] = torch.cos(i * pows)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))

        output = x + self.pe[:, :S]
        output = self.dropout(output)

        return output

# Loss for binary classification
# def loss_fn(outputs, labels, pos_weight=None):
#     """
#     Compute the cross entropy loss given outputs and labels.

#     Args:
#         outputs: (Variable) dimension batch_size x 6 - output of the model
#         labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

#     Returns:
#         loss (Variable): cross entropy loss for all images in the batch
#     """
#     loss = F.binary_cross_entropy_with_logits(outputs, labels,
#                                               pos_weight=pos_weight)
#     return loss



# Loss for multiclass classification
def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    """
    loss = F.cross_entropy(outputs, labels)
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) shape (N,1) - logits of the model
        labels: (np.ndarray) shape (N,1)  - ground truth

    Returns: (float) accuracy in [0,1]
    """
    pass


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    #'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
