# LSTM

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.embed import TimeFeatureEmbedding
from torch.nn import init


class LSTM_Encoder(nn.Module):

    def __init__(self, input_size, represent_size, e_layer, hidden_size=64, dropout=0.1):

        super(LSTM_Encoder, self).__init__()
        
        self.encoder_in = nn.LSTM(input_size, hidden_size, num_layers=e_layer, batch_first=True, dropout=dropout)
        self.encoder_out = nn.LSTM(hidden_size, represent_size, num_layers=1, batch_first=True, dropout=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        enc_out, _ = self.encoder_in(x)
        enc_out, _ = self.encoder_out(enc_out)
        enc_out = self.activation(enc_out)
        
        enc_out = enc_out.transpose(1, 2)
        
        return enc_out