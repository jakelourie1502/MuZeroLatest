from numpy.core.defchararray import upper
import torch
import numpy as np 
from global_settings import value_support, reward_support, value_only

def scalar_to_support_batch(value, min_support, max_support, support_levels):
    
    support_delim = (max_support-min_support) / (support_levels-1)
    lower_bound = torch.floor(value / support_delim)*support_delim
    upper_bound = lower_bound+support_delim
    lower_proportion = torch.round(10**3*((upper_bound - value) / support_delim)) / 10**3
    upper_proportion = 1-lower_proportion

    lower_idx = torch.round((lower_bound - min_support) / support_delim).long().squeeze()
    upper_idx = torch.where(lower_idx == support_levels-1,0,lower_idx + 1)
    lower_idx = torch.nn.functional.one_hot(lower_idx,support_levels)*lower_proportion
    upper_idx = torch.nn.functional.one_hot(upper_idx,support_levels)*upper_proportion
    support_vec = lower_idx+upper_idx

    return support_vec

def loss_func_r(r, true_values, dones_k, weights):
    """
    r is an M by value_support[2]
    true_values is a M by 1
    weigths is a M by 1
    dones_k is an M by 1
    """
    
    true_values = scalar_to_support_batch(true_values, *reward_support) #M by value_support[2]
    loss = -torch.sum(true_values * torch.log(r+1e-4),dim=1,keepdim=True)
    loss = weights * loss * (1-dones_k)
    return torch.mean(loss)

def loss_func_v(v, true_values, dones_k,weights):
    true_values = scalar_to_support_batch(true_values,*value_support)
    losses = -torch.sum(true_values * torch.log(v+1e-4),dim=1,keepdim=True)
    losses = weights*losses
    if not value_only:
        losses = losses * (1-dones_k)
    return torch.mean(losses)

def loss_func_p(p, true_policy, dones_k,weights):
    losses = torch.sum(true_policy * torch.log2(p + 1e-5),dim=1,keepdim=True)
    losses = -losses * weights * (1-dones_k)
    return torch.mean(losses)

def loss_func_entropy(p):
    return torch.mean(torch.sum(p * (torch.log2(p)+1e-3),dim=1))

def loss_func_proj(stopped_proj, w_grad_head, dones, weights):
    #L1 loss
    contrastive_loss = torch.sum(stopped_proj*w_grad_head,dim=1,keepdims=True)/(
            (torch.sum(stopped_proj**2,dim=1,keepdims=True)*torch.sum(w_grad_head**2,dim=1,keepdims=True))**0.5)
    return -torch.mean((1-dones)*weights*contrastive_loss)

class RDN_loss:
    def __init__(self, mu=0, siggy = 1):
        self.mu = mu
        self.siggy = siggy
        self.mu_sq = mu**2
        self.a = 0.01

    def evaluate(self,random_output, predicted_output):
        x = (torch.mean((random_output-predicted_output)**2)**0.5 - self.mu) / self.siggy
        return 1 / (1+np.exp(-0.5*x))

    def training(self, random_output, predicted_out):
        a = self.a
        mse = torch.mean((random_output-predicted_out)**2)
        self.mu = (1-a)*self.mu + a *(mse**0.5)
        self.mu_sq = (1-a)*self.mu_sq + a*mse
        self.siggy = (self.mu_sq - (self.mu**2)+1e-6)**0.5
        return mse
        