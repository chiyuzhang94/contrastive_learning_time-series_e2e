#TCN
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.embed import TimeFeatureEmbedding
from models.lstm_encoder import LSTM_Encoder
from models.dtcn_encoder import *
from models.lstm_encoder import LSTM_Encoder
from models.informer_encoder import Informer_Encoder
from models.tcn_encoder import *

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class TCN(nn.Module):
    def __init__(self, input_size, represent_size, output_size, e_layer, pred_leng, mask_rate, freq, model_name = "tcn", hidden_size=64,
                 kernel_size = 3, contrastive_loss=0.0, dropout = 0.1):
        super(TCN, self).__init__()
        print("Mask Rate:", mask_rate)
        print("CL loss lambda", contrastive_loss)
        
        self.contrastive_loss = contrastive_loss
        self.enc_embedding = nn.Linear(input_size, hidden_size)

        if 'lstm' in model_name:
            self.encoder = LSTM_Encoder(hidden_size, represent_size, e_layer, hidden_size)
        elif 'informer' in model_name:
            self.enc_embedding = nn.Linear(input_size, 128)
            self.encoder = Informer_Encoder(represent_size, d_model=128, e_layer = e_layer, d_ff=hidden_size, dropout=dropout)
        elif 'dtcn' in model_name:
            self.encoder = DilatedConvEncoder(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size)
        else:
            self.encoder = TemporalConvNet(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size, dropout=dropout)

        self.decoder = nn.Sequential(
            nn.Linear(represent_size, represent_size),
            nn.ReLU(),
            nn.Linear(represent_size, output_size)
        )

        self.decoder.apply(weights_init)
        
        self.drop = nn.Dropout(dropout)
        self.pred_leng = pred_leng
        self.mask_rate = mask_rate
        self.init_weights()

    def init_weights(self):
        self.enc_embedding.weight.data.normal_(0, 0.01)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x):
        enc_embed = self.drop(self.enc_embedding(x))

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out = self.encoder(enc_embed.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(enc_out)

        
        if self.contrastive_loss > 0.0:
            # masking augmentation
            mask = generate_binomial_mask(enc_embed.size(0), enc_embed.size(1), p=self.mask_rate).to(enc_embed.device)
            enc_embed_aug = enc_embed.clone()
            enc_embed_aug[~mask] = 0

            enc_out_aug = self.encoder(enc_embed_aug.transpose(1, 2)).transpose(1, 2)
        else:
            enc_out_aug = None

        return y[:,-self.pred_leng:,:], enc_out, enc_out_aug