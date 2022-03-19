import numpy as np
import torch

### ski and play
env_size = [8,82]
horizon_view = 16
observable_size = [8,16]
off_ramp = 20
start_state = env_size[1]*3+1
time_penalty = 0.004

goals = {(1,11):1}
dist = np.zeros((env_size[0]*env_size[1]+1))
dist[start_state] = 1 
max_plays = 250
play_random = 0
log_states = True
deque_length = 2

#### Model parameters
actions_size = 7
exploration_strategy = 'full' #none / immediate_only / full

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
reward_support = [-1,1,41]

#Prediction
res_block_pred = [32]
value_conv_channels = 32
value_hidden_dim = 64
value_output_supp = 1
policy_conv_channels = 32
policy_hidden_dim = 64
policy_output_supp = actions_size
value_support = [-1,1,41]
exp_v_support = [-10,10,101]

#SimSiam
proj_l1 = 64
proj_out = 64
pred_hid = 64

### RND
RND_output_vector = 64
rdn_beta = 1
#### Training parameters

#### mcts functions
c1 = 1
c2 = 19652
ucb_noise = [0,0.01]
temperature_init = 1
temperature_changes ={-1: 1, 512: 0.5}
play_limit_mcts = {-1:5, 30: 17, 60: 25,100: 33,400: 50}
manual_over_ride_play_limit = None    #only used in final testing - set to None otherwise
exponent_node_n = 1
ucb_denom_k = 1
use_policy = True
dirichlet_alpha = 1
N_steps = 5
mcts_update_mode = 'mean'
mcts_rolling_alpha = 0.1
bootstrap_start = 5

#### Main function
value_only = False
loading_in = False
start_epoch = 0
epochs = 10000
replay_buffer_size = 75000
gamma = 0.99
exp_gamma = 0.9

#### Training params
batch_size = 256
batches_per_train = 64
workers = 64
training_params = {'lr': 0.001,
                'lr_warmup': 25,
                'lr_decay': 0.25,
                'lr_decay_steps':1000,
                 'optimizer' : torch.optim.RMSprop,
                 'k': 4,
                 'value_coef': 0.25,
                 'siam_coef': 2,
                 'RDN_coef': 0.5,
                 'momentum' : 0.9,
                 'dones_coef': 0.5,
                 'future_nov_coef': 0.5,
                 'policy_ramp_up':1,
                 'l2': 0.0001,
                 'rho': 0.99 
                 }
epsilon_floor = 0.0
epsilon_ramp_epochs = 5
train_start_batch_multiple = 5
prioritised_replay = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pre_made_world = np.array(
            [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,.7,.7,.7,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,.7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1,1,.7,1,1,1,0,0,.7,.7,.7,1,1,0,0,0,0,1,1,1,1,1,1,.7,.7,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,.7,1,1,1,.7,.7,.7,.7,.7,1,1,1,1,.7,.7,.7,1,1,1,.7,.7,.7,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,.7,.7,.7,.7,.7,.7,.7,.7,.7,.7,.7,1,1,1,1,1,1,1,.7,1,1,1,1,1,1,1,.7,.7,.7,1,.7,.7,.7,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,.7,.7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,1,1,1,.7,.7,.7,.7,1,1,.7,.7,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,.7,.7,.7,1,.7,1,1,1,.7,1,.7,.7,1,.7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,1,1,.7,1,1,.7,.7,1,0,0,0,0,0,1,1,1,1,1,.7,.7,.7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,.7,.7,.7,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ],dtype='float')