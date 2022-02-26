import copy
import numpy as np
from .frozen_lake import gridWorld
import sys
sys.path.append("..")
sys.path.append(".")
from global_settings import gamma
from global_settings import env_size, goals, lakes, dist, actions_size, max_plays, play_random, lake_coverage,randomly_create_env, N_steps, value_support
from utils import vector2d_to_tensor_for_model
from .mcts import Node, MCTS
import torch
import time
 
class Episode:
    def __init__(self,model,epsilon):
        self.repr_model, self.dyn_model, self.prediction_model, self.rdn, self.rdn_pred = model.representation, model.dynamic, model.prediction, model.RDN, model.RDN_prediction
        self.epsilon = epsilon
        ### random goals
        self.env=gridWorld(env_size,lakes,goals, n_actions = actions_size, max_steps = max_plays, 
                        dist = dist, seed = None, rnd=play_random, lake_cov=lake_coverage,randomly_create_env= randomly_create_env)
        self.gamma = gamma
        
    
    def play_episode(self,RDN_OBJ, pick_best_policy=False, epoch=1,view_game = False):
        exploration_logger = np.zeros((env_size[0]*env_size[1]+1)) 
        running_reward = 0
        metrics = {}
        for met in ['action','obs','reward','done','policy','n_step_returns','V']:
            metrics[met] = []
        obs = self.env.reset()

        metrics['obs'].append(obs) #1 x 1 x obs(1) x obs(2)
        
        mcts = MCTS(episode = self,epoch = epoch, pick_best = pick_best_policy,RDN_OBJ = RDN_OBJ)
        
        q_current = 1
        while True:
            exploration_logger[self.env.state] +=1
            if view_game:
                
                self.env.render()
                time.sleep(0.2)
            
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.repr_model(obs.float())
            root_node = Node('null')
            root_node.state = state
            root_node.total_Q = q_current
            policy, action, root_node.total_Q, V = mcts.one_turn(root_node)
            # print(V)
            q_current = root_node.total_Q
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.randint(0,4)
            
            obs, _, reward, done = self.env.step(action)
            running_reward += reward
            self.store_metrics(action, reward, obs,metrics,done,policy, V)
            if done == True:
                break #params for ending episode
                
        self.calc_reward(metrics) #using N step returns or whatever to calculate the returns.
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        del obs; torch.cuda.empty_cache()
        # print("unit test: play_episode line 57 - printing reward, done, V and n step: ")
        # print(metrics['reward'])
        # print(metrics['done'])
        # print(metrics['V'])
        # print(metrics['n_step_returns'])
        # print(metrics['n_step_returns'])
        return metrics, running_reward, exploration_logger
        
    def store_metrics(self, action, reward,obs, metrics,done,policy, V):
        metrics['obs'].append(copy.deepcopy(obs))
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['done'].append(done)
        metrics['policy'].append(policy)
        metrics['V'].append(V)

    def calc_reward(self, metrics):
        Vs = metrics['V']
        Rs = np.array(metrics['reward'])
        Z = np.zeros_like((Rs))
        Z += Rs*gamma
        for n in range(1,N_steps):
            Z[:-n] += Rs[n:]*(gamma**(n+1))
        
        Z[:-N_steps] += np.array(Vs[N_steps:])
        Z = np.clip(Z,value_support[0],value_support[1])
        metrics['n_step_returns'] = Z

