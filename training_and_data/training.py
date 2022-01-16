import torch
import numpy

def loss_func_v(v, true_values, dones_k,weights):
    losses = (v - true_values)**2
    losses = weights*losses * (1-dones_k)
    return torch.mean(losses)

def loss_func_p(p, true_policy, dones_k,weights):
    losses = torch.sum(true_policy * torch.log2(p + 1e-5),dim=1)
    losses = -losses * weights * (1-dones_k)
    return torch.mean(losses)

def loss_func_entropy(p):
    return torch.mean(torch.sum(p * (torch.log2(p)+1e-3),dim=1))

    
