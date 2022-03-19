from threading import Thread
import time
import numpy as np
import os
from numpy.random import sample
import sys
from torch.optim import optimizer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from game_play.play_episode import Episode
import global_settings
from models.Representation import Representation
from models.Dynamic import Dynamic
from models.Prediction import Prediction
from models.JakeZero import JakeZero
from training_and_data.replay_buffer import Replay_Buffer
from training_and_data.training import  loss_func_v, loss_func_p, loss_func_r, loss_func_proj, RDN_loss,loss_func_future_nov
from global_settings import  exploration_strategy, log_states, exp_v_support, env_size, workers, epochs, device,  actions_size, value_only, training_params, batches_per_train, batch_size, loading_in, start_epoch, train_start_batch_multiple, prioritised_replay
import torch    
from utils import global_expl, get_epoch_params

np.set_printoptions(suppress=True)
sys.stdout = open("output_text.txt","w")

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

### Init misc
ts = time.time()
ep_history = []
training_started = False
pick_best = False
rdn_loss_obj = RDN_loss()
jake_zero.rdn_obj = rdn_loss_obj
if log_states: 
    explorer_log = global_expl()
### LOAD IN PARAMETERS
value_coef = training_params['value_coef']
siam_coef = training_params['siam_coef']
RDN_coef = training_params['RDN_coef']
dones_coef = training_params['dones_coef']
future_nov_coef = training_params['future_nov_coef']
K = training_params['k']


def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero_self_play,epsilon)
        metrics, rew,ep_explr_log = ep.play_episode(rdn_loss_obj, pick_best, epoch = e)
        replay_buffer.add_ep_log(metrics)
        if log_states: 
            explorer_log.append(ep_explr_log)
        ep_history.append(rew)

training_step_counter = start_epoch
for e in range(start_epoch, epochs):
    if training_started == True:
        training_step_counter += 1
    
    ## SETTING EPOCH SPECIFIC PARAMETERS
    epsilon, lr = get_epoch_params(e, training_step_counter)
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
    
    replay_buffer.purge() #keep replay buffer at reasonable size.
    rdn_loss_obj.update()
    ###### TRAINING
    ###unit testing
    siam_loss_log = []
    RDN_loss_log = []
    
    if len(replay_buffer.action_log) > batch_size*train_start_batch_multiple:
        training_started = True
        for batch in range(batches_per_train):
            
            ### getting target data
            sample_obs, indices, weights = replay_buffer.get_sample(prioritised_sampling=prioritised_replay)
            sample_obs = sample_obs.float()
            s = jake_zero.representation(sample_obs)
            done_tensor_tmin1 = torch.zeros((len(indices),K))
            done_tensor_same_t = torch.zeros((len(indices),K))
            weights = torch.tensor(weights).to(device).reshape(-1,1)
            
            loss = 0
            for k in range(K):
                action_index = np.array([replay_buffer.action_log[x+k] for x in indices])
                p, v, nov = jake_zero.prediction(s)
                s, r, d = jake_zero.dynamic(s,action_index)
                #note: episodic memory function can be done with dones last period because if it's dones this period, we still have an action to predict.

                ###### CAPTURING DONE MASKS (need two sets, as some variables require the prior k done, and some require this k done)
                ### SAME period
                dones_k_same_t = np.array([replay_buffer.done_logs[x+k] for x in indices])
                dones_k_same_t = torch.maximum(torch.tensor(dones_k_same_t), done_tensor_same_t[:, k-1]).to(device)
                done_tensor_same_t[:, k] = dones_k_same_t
                dones_k_same_t_in_format = dones_k_same_t.reshape(-1,1)
                

                ### PRIOR period
                if k == 0:
                    dones_k_tmin1 = done_tensor_tmin1[:, 0].to(device)
                else:
                    dones_k_tmin1 = np.array([replay_buffer.done_logs[x+k-1] for x in indices])
                    dones_k_tmin1 = torch.maximum(torch.tensor(dones_k_tmin1), done_tensor_tmin1[:, k-1]).to(device)
                    done_tensor_tmin1[:, k] = dones_k_tmin1
                dones_k_tmin1_in_format = dones_k_tmin1.reshape(-1,1)


                #### SIAM SECTION
                o = torch.tensor(np.array([replay_buffer.obs[x+k+1] for x in indices])).float()          
                with torch.no_grad():
                    reps = jake_zero.representation(o)
                    stopped_proj = jake_zero.project(reps, grad_branch = False)
                w_grad_head = jake_zero.project(s)
                loss_siam = loss_func_proj(stopped_proj, w_grad_head, dones_k_same_t_in_format,weights)
                loss += siam_coef * loss_siam
                siam_loss_log.append(float(loss_siam.detach().numpy())) ### UNIT testing cont'd for siam
                

                ## TERMINALITY
                loss_dones_k = -torch.mean(dones_k_same_t_in_format * torch.log(d+1e-4) + (1-dones_k_same_t_in_format)*torch.log(1-d+1e-4))
                loss+= dones_coef * loss_dones_k


                #### RDN SECTION
                if exploration_strategy != 'none':
                    RDN_random_output = jake_zero.RDN(s).detach()
                    RDN_prediction = jake_zero.RDN_prediction(s.detach())
                    loss_RDN_k = rdn_loss_obj.training(RDN_random_output,RDN_prediction,dones_k_tmin1_in_format, weights,batch+1+training_step_counter*batches_per_train,k)
                    RDN_loss_log.append(float(loss_RDN_k.detach().numpy()))
                    loss += RDN_coef *loss_RDN_k

                #### Future nov
                if exploration_strategy == 'full':
                    true_future_nov = torch.tensor(np.array([replay_buffer.future_exp_val[x+k] for x in indices])).to(device).reshape(-1, 1)
                    loss_future_novk = loss_func_future_nov(nov, true_future_nov, dones_k_tmin1_in_format, weights)
                    loss += future_nov_coef * loss_future_novk
                

                #### POLICY
                true_policy = torch.tensor(np.array([replay_buffer.policy_logs[x+k] for x in indices])).to(device).reshape(-1,actions_size)
                loss_Pk = loss_func_p(p, true_policy, dones_k_tmin1_in_format,weights)
                loss += loss_Pk

                #### VALUE AND REWARD (non-curiousity)
                if value_only:
                    true_values = torch.tensor(np.array([replay_buffer.n_step_returns[x] for x in indices])).to(device).reshape(-1,1) #here, when we just use value, we don't use the dones in the value calculation.
                else:
                    true_values = torch.tensor(np.array([replay_buffer.n_step_returns[x+k] for x in indices])).to(device).reshape(-1,1) 
                    true_rewards = torch.tensor(np.array([replay_buffer.reward_logs[x+k] for x in indices])).to(device).reshape(-1,1)
                    loss_Rk = loss_func_r(r, true_rewards, dones_k_tmin1_in_format, weights)
                    loss += loss_Rk
                loss_Vk = loss_func_v(v, true_values, dones_k_tmin1_in_format,weights)
                loss += loss_Vk * value_coef
                
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            
        ##### Save models and load to the self play models.
        jake_zero_self_play.load_state_dict(jake_zero.state_dict())
        torch.save(jake_zero, 'saved_models/jake_zero')
        if e % 10 ==0:
            print(e, ":", np.mean(ep_history[-300:]))
            if log_states: 
                print(explorer_log.log[:-1].reshape(env_size[0],env_size[1]))
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
                if exploration_strategy == 'full':
                    print("Average value from last batch of unclaimed novelty: ", torch.mean(true_future_nov))
                    nov = nov.detach().numpy()
                    nov_vals = np.linspace(*exp_v_support).reshape(1,-1)
                    nov_scalars = nov * nov_vals
                    print("Average value from last batch of predicted nov: ", np.mean(nov_scalars)*51)
            sys.stdout.flush()
            siam_loss_log = []
            RDN_loss_log = []