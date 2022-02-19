import numpy as np
import torch

### LAKE and play
env_size = (4,12)
goals = {(1,11):1}
dist = np.zeros((env_size[0]*env_size[1]+1))
dist[0] = 1 
lakes = [(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(1,2),(1,3),(1,4),(1,5),(2,7),(2,8),(2,9),(2,10),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)]
randomly_create_env = False
lake_coverage = [0.1,0.201]
max_plays = 50
play_random = 0
star = False

#### Model parameters
actions_size = 4
encodings_per_action = 8
#Repr
hidden_layer_rep_channels = 8
state_channels = 32
hidden_layer_rep_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
to_state_params = {'kernel_size': 1, 'stride': 1, 'padding': 0}
res_block_reps = [32] 
#Res
res_block_kernel_size = 3

#Dynamic
first_layer_dyn_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
res_block_dyns = [32] 
reward_conv_channels = 32
reward_hidden_dim = 64
reward_support = [0,1,11]

#Prediction
res_block_pred = [32]
value_conv_channels = 32
value_hidden_dim = 64
value_output_supp = 1
policy_conv_channels = 32
policy_hidden_dim = 64
policy_output_supp = 4
value_support = [0,1,11]

#SimSiam
proj_l1 = 128
proj_out = 128
pred_hid = 128
head_two_hidden = 8

### RND
RND_output_vector = 64
rdn_beta = 0.05
#### Training parameters

#### mcts functions
c1 = 1
c2 = 19652
ucb_noise = [0,0.01]
temperature_init = 1
temperature_changes ={-1: 1, 512: 0.5}
play_limit_mcts = {-1:5, 10: 17, 30: 25, 80: 38}
manual_over_ride_play_limit = 38    #only used in final testing - set to None otherwise
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
gamma = 1

#### Training params
batch_size = 256
batches_per_train = 64
workers = 64
training_params = {'lr': 0.002,
                'lr_warmup': 25,
                'lr_decay': 0.25,
                'lr_decay_steps':1000,
                 'optimizer' : torch.optim.RMSprop,
                 'k': 4,
                 'value_coef': 1,
                 'siam_coef': 2,
                 'RDN_coef': 0.5,
                 'momentum' : 0.9,
                 'dones_coef': 0.5,
                 'policy_ramp_up':1,
                 'entropy_coef': 0,
                 'entropy_first10': 1,
                 'l2': 0.0001,
                 'rho': 0.99 
                 }
epsilon_floor = 0.0
epsilon_ramp_epochs = 10
train_start_batch_multiple = 10
prioritised_replay = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
