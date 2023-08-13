import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.weight_norm import WeightNorm
from models.data_aug import Data_Aug, generate_binomial_mask
from typing import Union, Callable, Optional, List
import sys, math, random, copy
from models.embed import TimeFeatureEmbedding
from models.dtcn_encoder import *


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class DTCN(nn.Module):
    def __init__(self, input_size, represent_size, output_size, e_layer, pred_leng, mask_rate, freq, hidden_size=64,
                 kernel_size = 3, contrastive_loss=0.0, dropout = 0.1):
        super(DTCN, self).__init__()
        print("Mask Rate:", mask_rate)
        print("CL loss lambda", contrastive_loss)
        
        self.contrastive_loss = contrastive_loss
        self.enc_embedding = nn.Linear(input_size, hidden_size)
        self.tcn = DilatedConvEncoder(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size)

        self.decoder = nn.Sequential(
            nn.Linear(represent_size, represent_size),
            nn.ReLU(),
            nn.Linear(represent_size, output_size)
        )

        self.drop = nn.Dropout(dropout)
        self.pred_leng = pred_leng
        self.mask_rate = mask_rate

        self.init_weights()
        self.decoder.apply(weights_init)

    def init_weights(self):
        self.enc_embedding.weight.data.normal_(0, 0.01)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x):
        enc_embed = self.drop(self.enc_embedding(x))

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out = self.tcn(enc_embed.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(enc_out)

        
        if self.contrastive_loss > 0.0:
            # masking augmentation
            mask = generate_binomial_mask(enc_embed.size(0), enc_embed.size(1), p=self.mask_rate).to(enc_embed.device)
            enc_embed_aug = enc_embed.clone()
            enc_embed_aug[~mask] = 0

            enc_out_aug = self.tcn(enc_embed_aug.transpose(1, 2)).transpose(1, 2)
        else:
            enc_out_aug = None


        return y[:,-self.pred_leng:,:], enc_out, enc_out_aug