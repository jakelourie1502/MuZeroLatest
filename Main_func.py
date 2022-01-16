

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
from training_and_data.training import loss_func_v, loss_func_p, loss_func_entropy
from global_settings import workers, gamma, epochs, device, gamma, actions_size, training_params, batches_per_train, batch_size, loading_in, start_epoch, train_start_batch_multiple, prioritised_replay
import torch    
from numpy import save
import gc
import torch.nn.functional as TF
from utils import get_epoch_params
from pympler import asizeof


### LOAD IN PARAMETERS
batch_size = global_settings.batch_size
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

def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero_self_play,epsilon)
        metrics, rew = ep.play_episode(pick_best, epoch = e)
        replay_buffer.add_ep_log(metrics)
        ep_history.append(rew)

training_step_counter = start_epoch
for e in range(start_epoch, epochs):
    if training_started == True:
        training_step_counter += 1
    
    ## SETTING EPOCH SPECIFIC PARAMETERS
    policy_coef, entropy_coef, epsilon, pick_best, lr = get_epoch_params(e, training_step_counter)
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
    if len(replay_buffer.action_log) > batch_size*train_start_batch_multiple:
        training_started = True
        for _ in range(batches_per_train):
            
            ### getting target data
            sample_obs, indices, weights = replay_buffer.get_sample(prioritised_sampling=prioritised_replay)
            sample_obs = sample_obs.float()
            s = jake_zero.representation(sample_obs)
            done_tensor = torch.zeros((len(indices),K))
            done_pol_tensor = torch.zeros((len(indices),K))
            weights = torch.tensor(weights).to(device).reshape(-1,1)
            
            loss = 0
            for k in range(K):
                action_index = np.array([replay_buffer.action_log[x+k] for x in indices])
                s = jake_zero.dynamic(s,action_index)
                p, v = jake_zero.prediction(s)
                
                ### get the done masks for the value targets (when in goal / lake state, a move should return a value, whereas it should not return a policy. when we have reward, V done mask should behave like P done mask)
                dones = np.array([replay_buffer.done_logs[x+k-1] for x in indices])
                if k == 0:
                    dones_k = done_tensor[:, 0].to(device)
                else:
                    dones_k = torch.maximum(torch.tensor(dones), done_tensor[:, k-1]).to(device)
                    done_tensor[:, k] = dones_k
                
                ### get the done masks for the policy targets
                dones_pol_k = np.array([replay_buffer.done_logs[x+k] for x in indices])
                dones_pol_k = torch.maximum(torch.tensor(dones_pol_k), done_pol_tensor[:, k-1]).to(device)
                done_pol_tensor[:, k] = dones_pol_k

                true_values = torch.tensor(np.array([replay_buffer.fut_val_logs[x+k] for x in indices])).to(device).reshape(-1,1)
                true_policy = torch.tensor(np.array([replay_buffer.policy_logs[x+k+1] for x in indices])).to(device).reshape(-1,actions_size)
                loss_Vk = loss_func_v(v, true_values, dones_k.reshape(-1,1),weights)
                loss_Pk = loss_func_p(p, true_policy, dones_pol_k.reshape(-1,1),weights)
                loss_entk = loss_func_entropy(p)
                loss += loss_Vk
                loss += loss_Pk * policy_coef
                loss += loss_entk * entropy_coef
                # print(true_policy, p)
            #### UNIT TEST IN LINE
            # print("Line 134 checkRatio of V to P to E should be roughly 10: 2: 1:\n ", loss_Vk, policy_coef * loss_Pk, entropy_coef * loss_entk)
            # print("Line 134 check: when we print the two done matrices, the policy one (second) should look like the first one one moved one to the left")
            # print(done_tensor)
            # print(done_pol_tensor)
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
        
        ##### Save models and load to the self play models.
        if e % 10 ==0:
            print(e, ":", np.mean(ep_history[-300:]))
            if e % 25 == 0:
                rew_test = []
                for _ in range(100):
                    with torch.no_grad():
                        ep = Episode(jake_zero_self_play,0)
                        _, rew = ep.play_episode(True, epoch = max(51,e))
                        rew_test.append(rew)
                print("Test reward: ", np.mean(rew_test))
                print("LR: ", lr)
                print("replay buffer size: ", len(replay_buffer.done_logs))
                print("training steps: ", training_step_counter)
            jake_zero_self_play.load_state_dict(jake_zero.state_dict())
            torch.save(jake_zero, 'saved_models/jake_zero')
            