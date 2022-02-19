

from threading import Thread
import time
import numpy as np
import os
from numpy.random import sample

from torch.optim import optimizer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from game_play.frozen_lake import gridWorld
from game_play.play_episode import Episode
import global_settings
from models.Representation import Representation
from models.Dynamic import Dynamic
from models.Prediction import Prediction
from models.JakeZero import JakeZero
from training_and_data.replay_buffer import Replay_Buffer
from training_and_data.training import loss_func_v, loss_func_p, loss_func_entropy, loss_func_r, loss_func_proj, RDN_loss
from global_settings import  env_size, workers, gamma, epochs, device, gamma, actions_size, value_only, training_params, batches_per_train, batch_size, loading_in, start_epoch, train_start_batch_multiple, prioritised_replay
import torch    
from numpy import save
import gc
import torch.nn.functional as TF
from utils import global_expl, get_epoch_params
from pympler import asizeof
import sys
np.set_printoptions(suppress=True)
sys.stdout = open("output_text.txt","w")

### LOAD IN PARAMETERS
K = training_params['k']

### INITIALISE MODELS

### Create 2 models pre model (self_play and train)

repr, repr_target = Representation().to(device), Representation().to(device)
dynamic, dynamic_target = Dynamic().to(device), Dynamic().to(device)
pred_model, pred_model_target = Prediction().to(device), Prediction().to(device)
jake_zero = JakeZero(repr, dynamic, pred_model).to(device)
jake_zero.train()
#### IF LOADING IN == TRUE
if loading_in:
    print('loaded in')
    jake_zero = torch.load('saved_models/jake_zero')

jake_zero_self_play = JakeZero(repr_target, dynamic_target, pred_model_target).to(device)
jake_zero_self_play.load_state_dict(jake_zero.state_dict())
jake_zero_self_play.eval()

####### INITIALISE OPTIMIZER
optimizer = training_params['optimizer'](jake_zero.parameters(), lr=training_params['lr'], momentum = training_params['momentum'], alpha = training_params['rho'],weight_decay = training_params['l2'])
replay_buffer = Replay_Buffer()

#### MISC
ts = time.time()
ep_history = []
training_started = False
pick_best = False
value_coef = training_params['value_coef']
siam_coef = training_params['siam_coef']
RDN_coef = training_params['RDN_coef']
dones_coef = training_params['dones_coef']
explorer_log = global_expl()
rdn_loss_obj = RDN_loss()
jake_zero.rdn_obj = rdn_loss_obj

def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero_self_play,epsilon)
        metrics, rew,ep_explr_log = ep.play_episode(rdn_loss_obj, pick_best, epoch = e)
        replay_buffer.add_ep_log(metrics)
        explorer_log.append(ep_explr_log)
        ep_history.append(rew)

training_step_counter = start_epoch
for e in range(start_epoch, epochs):
    if training_started == True:
        training_step_counter += 1
    
    ## SETTING EPOCH SPECIFIC PARAMETERS
    entropy_coef, epsilon, lr = get_epoch_params(e, training_step_counter)
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    ##### Threading
    t = []
    for _ in range(workers):
        thread = Thread(target=Play_Episode_Wrapper, args=())
        t.append(thread)
        thread.start()
    for thread in t:
        thread.join()
        del thread
        torch.cuda.empty_cache()
    
    replay_buffer.purge() #keep replay buffer at reasonable size.

    ###### TRAINING
    #unit testing
    siam_loss_log = []
    RDN_loss_log = []
    if len(replay_buffer.action_log) > batch_size*train_start_batch_multiple:
        training_started = True
        for _ in range(batches_per_train):
            
            ### getting target data
            sample_obs, indices, weights = replay_buffer.get_sample(prioritised_sampling=prioritised_replay)
            sample_obs = sample_obs.float()
            s = jake_zero.representation(sample_obs)
            done_tensor = torch.zeros((len(indices),K))
            done_tensor_siam = torch.zeros((len(indices),K))
            weights = torch.tensor(weights).to(device).reshape(-1,1)
            
            loss = 0
            for k in range(K):
                action_index = np.array([replay_buffer.action_log[x+k] for x in indices])
                p, v = jake_zero.prediction(s)
                s, r, d = jake_zero.dynamic(s,action_index)
                
                #### SIAM SECTION
                o = torch.tensor(np.array([replay_buffer.obs[x+k+1] for x in indices])).float()          
                # needs it own dones
                dones_siam = np.array([replay_buffer.done_logs[x+k] for x in indices])
                dones_k_siam = torch.maximum(torch.tensor(dones_siam), done_tensor_siam[:, k-1]).to(device)
                done_tensor_siam[:, k] = dones_k_siam
                dones_k_siam_in_format = dones_k_siam.reshape(-1,1)
                
                ## get two things for loss
                with torch.no_grad():
                    reps = jake_zero.representation(o)
                    stopped_proj = jake_zero.project(reps, grad_branch = False)
                w_grad_head = jake_zero.project(s)
                loss_siam = loss_func_proj(stopped_proj, w_grad_head, dones_k_siam_in_format,weights)
                siam_loss_log.append(float(loss_siam.detach().numpy()))
                
                ## done - uses siam dones_k because it's the same time stamp.
                # print("unit test: shape of d and of dones_k_siam should both be batch, 1")
                # print(d.shape, dones_k_siam.reshape(-1,1).shape)
                # print(dones_k_siam_in_format, d)
                # sys.stdout.flush()
                loss_dones_k = -torch.mean(dones_k_siam_in_format * torch.log(d+1e-4) + (1-dones_k_siam_in_format)*torch.log(1-d+1e-4))
                loss+= dones_coef * loss_dones_k

                #### RDN SECTION
                RDN_random_output = jake_zero.RDN(s).detach()
                RDN_prediction = jake_zero.RDN_prediction(s)
                loss_RDN_k = rdn_loss_obj.training(RDN_random_output,RDN_prediction)
                RDN_loss_log.append(float(loss_RDN_k.detach().numpy()))
                loss += RDN_coef *loss_RDN_k

                #### MAIN VALUE AND POLICY SECTION
                dones = np.array([replay_buffer.done_logs[x+k-1] for x in indices])
                if k == 0:
                    dones_k = done_tensor[:, 0].to(device)
                else:
                    dones_k = torch.maximum(torch.tensor(dones), done_tensor[:, k-1]).to(device)
                    done_tensor[:, k] = dones_k
                
                dones_in_format = dones_k.reshape(-1,1)
                true_policy = torch.tensor(np.array([replay_buffer.policy_logs[x+k] for x in indices])).to(device).reshape(-1,actions_size)
                if value_only:
                    true_values = torch.tensor(np.array([replay_buffer.n_step_returns[x] for x in indices])).to(device).reshape(-1,1) #here, when we just use value, we don't use the dones in the value calculation.
                else:
                    true_values = torch.tensor(np.array([replay_buffer.n_step_returns[x+k] for x in indices])).to(device).reshape(-1,1) 
                    true_rewards = torch.tensor(np.array([replay_buffer.reward_logs[x+k] for x in indices])).to(device).reshape(-1,1)
                    loss_Rk = loss_func_r(r, true_rewards, dones_in_format, weights)
                    loss += loss_Rk
                loss_Vk = loss_func_v(v, true_values, dones_in_format,weights)
                loss_Pk = loss_func_p(p, true_policy, dones_in_format,weights)
                loss_entk = loss_func_entropy(p)
                loss += loss_Vk * value_coef
                loss += loss_Pk
                loss += loss_entk * entropy_coef
                loss += siam_coef * loss_siam
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            
        ##### Save models and load to the self play models.
        jake_zero_self_play.load_state_dict(jake_zero.state_dict())
        torch.save(jake_zero, 'saved_models/jake_zero')
        if e % 10 ==0:
            print(e, ":", np.mean(ep_history[-300:]))
            print(explorer_log.log[:-1].reshape(env_size[0],env_size[1]))
            print(rdn_loss_obj.mu, rdn_loss_obj.siggy)
            if e % 10 == 0:
                rew_test = []
                for _ in range(100):
                    with torch.no_grad():
                        ep = Episode(jake_zero_self_play,0)
                        _, rew,_ = ep.play_episode(rdn_loss_obj, True, epoch = max(51,e))
                        rew_test.append(rew)
                print("Test reward: ", np.mean(rew_test))
                print("LR: ", lr)
                print("replay buffer size: ", len(replay_buffer.done_logs))
                print("training steps: ", training_step_counter)
                print("Average siam loss: ", np.mean(siam_loss_log))
                print("Average RDN loss: ", np.mean(RDN_loss_log))
            jake_zero_self_play.load_state_dict(jake_zero.state_dict())
            torch.save(jake_zero, 'saved_models/jake_zero')
            sys.stdout.flush()
            siam_loss_log = []
            RDN_loss_log = []