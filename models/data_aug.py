import sys, math, random, copy
import torch
import numpy as np
import torch

class Data_Aug():
    def __init__(self,
                 sigma=0.5,
                 p=0.5):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def cost_transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma).to(x.device)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1).to(x.device)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma).to(x.device)

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)