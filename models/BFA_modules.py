import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

class Bilinear_Activation_slice(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers):
        super(Bilinear_Activation, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_pairs = num_pairs
        self.lin_s = nn.Linear(input_dim, out_dim)
        self.lin_q = nn.Linear(input_dim, out_dim)
#         self.lin_o = nn.Linear(out_dim, input_dim)
        fs = []
        for i in range(self.num_pairs):
            drop = nn.Dropout(p=0.5)
            squeeze = nn.Linear(out_dim, hidden_dim)
            fs.append(nn.Sequential(drop, squeeze))
            
        self.supportvec_layers  = nn.ModuleList(fs)
        
        fq = []
        for i in range(self.num_pairs):
            drop = nn.Dropout(p=0.5)
            squeeze = nn.Linear(out_dim, hidden_dim)
            fq.append(nn.Sequential(do, squeeze))
        self.queryvec_layers  = nn.ModuleList(fq)

    def forward(self, query_emb, support_emb):
        batch_size = support_emb.size()[0]
        f_fuse  = []
        que_emb = self.lin_q(query_emb.transpose(1,0))
        sup_emb = self.lin_s(support_emb.transpose(1,0))
        
        for i in range(self.num_pairs):
            f_fs = self.supportvec_layers[i](sup_emb)
            f_fq = self.queryvec_layers[i](que_emb)
            f_fuse.append(torch.mm(f_fq, f_fs.transpose(1,0)))
        f_fuse = torch.stack(f_fuse, dim=1) 
        f_fuse = f_fuse.sum(1)
#         f_fuse = f_fuse.sum(1)
#         f_fuse = F.sigmoid(f_fuse)
        return f_fuse
    
    
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
