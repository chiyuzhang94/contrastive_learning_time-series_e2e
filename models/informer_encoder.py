import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class Informer_Encoder(nn.Module):
    def __init__(self, represent_size, 
                factor=3, d_model=128, n_heads=8, e_layer=5, d_ff=64, 
                dropout=0.1, attn='prob', activation='gelu', 
                output_attention = False, distil=False):
        super(Informer_Encoder, self).__init__()

        self.attn = attn
        self.output_attention = output_attention

        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layer)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layer-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, represent_size, bias=True)
        
    def forward(self, x, enc_self_mask=None):

        x = x.transpose(1, 2)
        enc_out, _ = self.encoder(x, attn_mask=enc_self_mask)

        # print("dec_out", dec_out.shape)
        enc_out = self.projection(enc_out)

        enc_out = enc_out.transpose(1, 2)
        
        return enc_out