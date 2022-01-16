import torch
import numpy as np
from global_settings import device, epsilon_ramp_epochs, epsilon_floor, training_params

def vector2d_to_tensor_for_model(state):
    return torch.unsqueeze(torch.tensor(state),0).to(device)

def get_epoch_params(e,training_step_counter):
    
    #### SET policy and entropy coef
    if e > training_params['policy_ramp_up']:
        policy_coef = training_params['policy_coef']
    else:
        policy_coef = training_params['policy_coef'] * (e/ training_params['policy_ramp_up'])
    if training_step_counter < 10:
        entropy_coef = training_params['entropy_first10']
    else:
        entropy_coef = training_params['entropy_coef']
    
    #Set epsilon
    epsilon = max(epsilon_floor, 1-(e/epsilon_ramp_epochs))
    pick_best = False
    
    if training_step_counter < training_params['lr_warmup']:
        lr = training_params['lr'] * training_step_counter / training_params['lr_warmup']
    else:
        lr = training_params['lr'] * training_params['lr_decay'] ** ((training_step_counter - training_params['lr_warmup']) // training_params['lr_decay_steps'])
    return policy_coef, entropy_coef, epsilon, pick_best, lr