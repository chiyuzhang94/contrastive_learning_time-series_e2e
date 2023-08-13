import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.weight_norm import WeightNorm
from models.data_aug import Data_Aug
from typing import Union, Callable, Optional, List
import sys, math, random, copy
from models.embed import TimeFeatureEmbedding
from models.dtcn_encoder import *
from models.embed import DataEmbedding
from einops import reduce, rearrange, repeat



class Dialated_TCNBase(nn.Module):
    def __init__(self, input_size, represent_size, e_layer, freq, hidden_size = 64,
                 kernel_size = 3, dropout = 0.1, mare=False, time_feature_embed = False, embed='timeF'):
        super(Dialated_TCNBase, self).__init__()
        
        self.mare = mare
        self.time_feature_embed = time_feature_embed

        if self.time_feature_embed:
            self.enc_embedding = DataEmbedding(input_size, hidden_size, embed_type = embed, freq=freq) ## embedding method from Infomer
        else:
            # self.contrastive_loss = contrastive_loss
            self.enc_embedding = nn.Linear(input_size, hidden_size)
            self.init_weights()

        self.tcn = DilatedConvEncoder(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size)

        if self.mare:
            self.kernels = [1, 2, 4, 8, 16, 32, 64, 128]
            self.tfd = nn.ModuleList(
                [nn.Conv1d(represent_size, represent_size, k, padding=k-1) for k in self.kernels]
            )

        self.drop = nn.Dropout(dropout)

    def init_weights(self):
        self.enc_embedding.weight.data.normal_(0, 0.01)

    def forward(self, x, x_mark):
        
        if self.time_feature_embed:
            enc_embed = self.drop(self.enc_embedding(x, x_mark))
        else: 
            enc_embed = self.drop(self.enc_embedding(x))

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out = self.tcn(enc_embed.transpose(1, 2)).transpose(1, 2)

        if self.mare:
            enc_out = enc_out.transpose(1, 2)
            # print("enc_out before", enc_out.shape)
            trend = []
            for idx, mod in enumerate(self.tfd):
                out = mod(enc_out)  # b d t
                if self.kernels[idx] != 1:
                    out = out[..., :-(self.kernels[idx] - 1)]
                trend.append(out.transpose(1, 2))  # b t d
            
            enc_out = reduce(
                rearrange(trend, 'list b t d -> list b t d'),
                'list b t d -> b t d', 'mean'
            )
            # print("enc_out", enc_out.shape)

        return self.drop(enc_out)

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class DTCN_MoCo(nn.Module):
    def __init__(self, 
                 input_size, hidden_size, output_size, e_layer, pred_leng, freq,
                 kernel_size = 3, dropout = 0.1, l2norm = False, average_pool = True, data_aug = None, mare = False, time_feature_embed = False, embed='timeF',
                 device: Optional[str] = 'cuda',
                 K: Optional[int] = 256,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 1.0):
        super(DTCN_MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.l2norm = l2norm
        self.average_pool = average_pool
        self.data_aug = data_aug

        if self.data_aug == "cost":
            print("Use Data augmentation method:", self.data_aug)
            self.data_aug_tool = Data_Aug(sigma=0.5, p=0.5)
        else:
            self.data_aug = False

        print("l2norm", self.l2norm )

        self.encoder_q = Dialated_TCNBase(input_size, hidden_size, e_layer, freq, kernel_size = kernel_size, dropout = dropout, mare = mare, time_feature_embed = time_feature_embed, embed=embed)
        self.encoder_k = copy.deepcopy(self.encoder_q)


        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.head_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(hidden_size, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.decoder.apply(weights_init)
        
        self.drop = nn.Dropout(dropout)
        self.pred_leng = pred_leng
        # self.init_weights()

    def compute_loss(self, q, k, k_negs):
        
        if self.l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            k_negs = F.normalize(k_negs, dim=-1)
            
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, x, x_mark):
        enc_out = self.encoder_q(x, x_mark)
        
        if enc_out is not None:
            if self.average_pool:
                q_t = self.head_q(torch.mean(enc_out, 1))
            else:
                rand_idx = np.random.randint(0, x.shape[1])
                q_t = self.head_q(enc_out[:,rand_idx, :])

        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder

            if self.data_aug:
                x_k = self.data_aug_tool.cost_transform(x)
            else:
                x_k =x

            k_t = self.encoder_k(x_k, x_mark)
            
            if k_t is not None:
                if self.average_pool: 
                    k_t = self.head_k(torch.mean(k_t, 1))
                else:
                    k_t = self.head_k(k_t[:,rand_idx, :])

        moco_loss = self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        y = self.decoder(enc_out)
        
        return y[:,-self.pred_leng:,:], enc_out, moco_loss
