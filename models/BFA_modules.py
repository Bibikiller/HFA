import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
    
    
class Diffusion(nn.Module):
    def __init__(self, max_step=None):
        super(Diffusion, self).__init__()
        self.max_step = max_step

    def forward(self, input, weight):
        n, c, h, w = input.size()
        weight = weight.view(n, c, 9, h * w)
        weight = torch.abs(weight) / torch.sum(torch.abs(weight), dim=2).unsqueeze(2)

        x = input
        for i in range(min(self.max_step, max(h, w))):
            x = F.unfold(x, kernel_size=3, padding=1).view(n, c, 9, h * w)
            x = (x * weight).sum(2).view(n, c, h, w)
        return x
