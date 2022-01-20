import numpy as np
import torch

### LAKE and play
env_size = (4,4)
goals = {(0,0):1}
dist = np.zeros((env_size[0]*env_size[1]+1))
dist[0] = 1 #obsolete if using random_start
randomly_create_env = True
lakes = []
lake_coverage = [0.1,0.201]
max_plays = 15
play_random = 0

#### Model parameters
actions_size = 4
encodings_per_action = 8
#Repr
hidden_layer_rep_channels = 8
state_channels = 32
hidden_layer_rep_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
to_state_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
res_block_reps = [32] #relates to downsampling
#Res
res_block_kernel_size = 3

#Dynamic
first_layer_dyn_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
res_block_dyns = [32] #relates to downsampling
reward_conv_channels = 32
reward_hidden_dim = 64
reward_support = [-1,1,21]

#Prediction
res_block_pred = [32]
value_conv_channels = 32
value_hidden_dim = 64
value_output_supp = 1
policy_conv_channels = 32
policy_hidden_dim = 64
policy_output_supp = 4
value_support = [-2,2,41]

#### Training parameters

#### mcts functions
c1 = 1
c2 = 19652
ucb_noise = [0,0.01]
temperature_init = 1
temperature_changes ={-1: 1, 512: 0.5}
play_limit_mcts = {-1:5, 30: 13, 128: 25}
manual_over_ride_play_limit = None #only used in final testing
exponent_node_n = 1
ucb_denom_k = 1
use_policy = True
dirichlet_alpha = 1
N_steps = 5

#### Main function
value_only = False
loading_in = False
start_epoch = 0
epochs = 10000
replay_buffer_size = 75000
gamma = 0.99

#### Training params
batch_size = 256
batches_per_train = 32
workers = 64
training_params = {'lr': 0.002,
                'lr_warmup': 25,
                'lr_decay': 0.25,
                'lr_decay_steps':1000,
                 'optimizer' : torch.optim.RMSprop,
                 'k': 4,
                 'value_coef': 0.25,
                 'momentum' : 0.9,
                 'policy_ramp_up':1,
                 'entropy_coef': 0,
                 'entropy_first10': 1,
                 'l2': 0.0001,
                 'rho': 0.99 
                 }
epsilon_floor = 0.0
epsilon_ramp_epochs = 100
train_start_batch_multiple = 10
prioritised_replay = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
