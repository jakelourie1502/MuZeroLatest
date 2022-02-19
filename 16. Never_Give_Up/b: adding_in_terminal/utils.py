import torch
import numpy as np
from global_settings import device, epsilon_ramp_epochs, epsilon_floor, training_params, env_size

def vector2d_to_tensor_for_model(state):
    return torch.unsqueeze(torch.tensor(state),0).to(device)

def get_epoch_params(e,training_step_counter):
    
    #### SET policy and entropy coef
    if training_step_counter < 10:
        entropy_coef = training_params['entropy_first10']
    else:
        entropy_coef = training_params['entropy_coef']
    
    #Set epsilon
    epsilon = max(epsilon_floor, 1-(e/epsilon_ramp_epochs))
    
    if training_step_counter < training_params['lr_warmup']:
        lr = training_params['lr'] * training_step_counter / training_params['lr_warmup']
    else:
        lr = training_params['lr'] * training_params['lr_decay'] ** ((training_step_counter - training_params['lr_warmup']) // training_params['lr_decay_steps'])
    return entropy_coef, epsilon, lr

def scalar_to_support(value, min_support, max_support, support_levels):
    """
    Note: takes numpy arrays
    in: requires scalar value, not array of one scalar value.
    outputs 1 dimensional vector
    """
    support_vals =  np.linspace(min_support, max_support, support_levels)
    support_vec = np.zeros((support_levels))
    if value in support_vals:
        idx = np.where(support_vals == value)
        support_vec[idx] = 1
    else:
        support_delim = (max_support-min_support) / (support_levels-1)

        lower_bound = (value // support_delim)*support_delim
        upper_bound = lower_bound+support_delim
        lower_proportion = round((upper_bound - value) / support_delim,5)
        upper_proportion = 1-lower_proportion
        lower_idx = int(round((lower_bound - min_support) / support_delim,1))
        upper_idx = lower_idx +1
        support_vec[lower_idx] = lower_proportion
        support_vec[upper_idx] = upper_proportion
    return support_vec

def support_to_scalar(array, min_support, max_support, support_levels):
    """
    in: 1d vector
    Note: takes numpy arrays
    outputs scalar
    """
    support_vals = np.linspace(min_support, max_support, support_levels)
    return round(np.sum(support_vals * array),4)

class global_expl:
    def __init__(self):
        self.log = np.zeros((env_size[0]*env_size[1]+1))
    def append(self, log):
        self.log += log