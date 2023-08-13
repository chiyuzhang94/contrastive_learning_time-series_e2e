## reuse code from https://github.com/salesforce/C
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from models.weight_norm import WeightNorm
from models.data_aug import Data_Aug, generate_binomial_mask
from typing import Union, Callable, Optional, List
import sys, math, random, copy
# from models.embed import TimeFeatureEmbedding
from models.dtcn_encoder import *
import torch.fft as fft
from einops import reduce, rearrange, repeat

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        # nan_mask = ~x.isnan().any(axis=-1)
        # x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        # mask &= nan_mask
        x[~mask] = 0

        x = self.feature_extractor(x.transpose(1, 2))  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d

        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        season = []
        for mod in self.sfd:
            out = mod(x.transpose(1, 2))  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)


class COST_E2E(nn.Module):
    def __init__(self, 
                 input_size, hidden_size, output_size, e_layer, pred_leng, input_length,
                  dropout = 0.1, alpha = 0.0005, l2norm = True, data_aug = "cost",
                 device: Optional[str] = 'cuda',
                 K: Optional[int] = 256,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 1.0):
        super(COST_E2E, self).__init__()
        
        self.alpha= alpha
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.l2norm = l2norm
        self.data_aug = data_aug

        if self.data_aug == "cost":
            print("Use Data augmentation method:", self.data_aug)
            self.data_aug_tool = Data_Aug(sigma=0.5, p=0.5)

        print("l2norm", self.l2norm )

        self.encoder_q = CoSTEncoder(input_size, output_dims = hidden_size, 
                                     kernels = [1, 2, 4, 8, 16, 32, 64, 128],
                                     length = input_length, depth = e_layer)

        self.encoder_k = copy.deepcopy(self.encoder_q)


        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims),
            nn.ReLU(),
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims)
        )
        self.head_k = nn.Sequential(
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims),
            nn.ReLU(),
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(self.encoder_q.component_dims, K), dim=0))
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

    # def init_weights(self):
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.normal_(0, 0.01)

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

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

    def forward(self, x):
        # compute query features
        x_q, x_k = x, x
        rand_idx = np.random.randint(0, x.shape[1])

        q_t, q_s = self.encoder_q(x_q, mask='all_true')
        enc_out = torch.cat([q_t, q_s], dim=-1)

        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)
        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder

            if self.data_aug:
                x_k = self.data_aug_tool.cost_transform(x_k)

            k_t, _ = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)

        loss = 0

        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k)
        k_s = F.normalize(k_s, dim=-1)

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        
        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        loss += (self.alpha * (seasonal_loss/2))

        y = self.decoder(enc_out)
        
        return y[:,-self.pred_leng:,:], enc_out, loss


