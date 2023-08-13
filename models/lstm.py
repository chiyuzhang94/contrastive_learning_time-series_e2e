# LSTM

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.embed import TimeFeatureEmbedding
from torch.nn import init
from models.lstm_encoder import LSTM_Encoder


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class LSTM(nn.Module):

    def __init__(self, input_size, represent_size, output_size, e_layer, pred_leng, mask_rate, freq, hidden_size=64,
                 contrastive_loss=0.0, dropout=0.1):

        super(LSTM, self).__init__()
        print("Mask Rate:", mask_rate)
        print("CL loss lambda", contrastive_loss)
        
        self.contrastive_loss = contrastive_loss
        self.enc_embedding = nn.Linear(input_size, hidden_size)

        
        self.encoder = LSTM_Encoder(hidden_size, represent_size, e_layer, hidden_size)

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

    def forward(self, x):
        # print("x", x.shape)
        enc_embed = self.drop(self.enc_embedding(x))
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out = self.encoder(enc_embed.transpose(1, 2)).transpose(1, 2) 
        
        y = self.decoder(enc_out)

        # enc_out_aug = None
        if self.contrastive_loss > 0.0:
            # masking augmentation
            mask = generate_binomial_mask(enc_embed.size(0), enc_embed.size(1), p=self.mask_rate).to(enc_embed.device)
            enc_embed_aug = enc_embed.clone()
            enc_embed_aug[~mask] = 0
            enc_out_aug = self.encoder(enc_embed_aug.transpose(1, 2)).transpose(1, 2) 
        else:
            enc_out_aug = None


        return y[:,-self.pred_leng:,:], enc_out, enc_out_aug