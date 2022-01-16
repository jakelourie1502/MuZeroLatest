import copy
import numpy as np
from .frozen_lake import gridWorld
import sys
sys.path.append("..")
sys.path.append(".")
from global_settings import gamma
from global_settings import env_size, goals, lakes, dist, actions_size, max_plays, play_random, lake_coverage,randomly_create_env
from utils import vector2d_to_tensor_for_model
from .mcts import Node, MCTS
import torch
import time
 
class Episode:
    def __init__(self,model,epsilon):
        self.repr_model, self.dyn_model, self.prediction_model = model.representation, model.dynamic, model.prediction
        self.epsilon = epsilon
        ### random goals
        self.env=gridWorld(env_size,lakes,goals, n_actions = actions_size, max_steps = max_plays, 
                        dist = dist, seed = None, rnd=play_random, lake_cov=lake_coverage,randomly_create_env= randomly_create_env)
        self.gamma = gamma
        
    
    def play_episode(self,pick_best_policy=False, epoch=1,view_game = False):
        metrics = {}
        for met in ['action','obs','reward','done','policy','V','search_val_logs']:
            metrics[met] = []
        obs = self.env.reset()

        metrics['obs'].append(obs) #1 x 1 x obs(1) x obs(2)
        mcts = MCTS(episode = self,epoch = epoch, pick_best = pick_best_policy)
        
        q_current = 1
        while True:
            if view_game:
                self.env.render()
                time.sleep(0.2)
            
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.repr_model(obs.float())
            root_node = Node('null',state)
            root_node.Q = q_current
            policy, action, root_node.Q = mcts.one_turn(root_node)
            q_current = root_node.Q
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.randint(0,4)
            
            obs, _, reward, done = self.env.step(action)
            self.store_metrics(action, reward, obs,metrics,done,policy, root_node.Q)
            if done == True:
                break #params for ending episode
                
        self.calculate_V_from_rewards(metrics) #using N step returns or whatever to calculate the returns.
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        del obs; torch.cuda.empty_cache()
        return metrics, reward
        
    def store_metrics(self, action, reward,obs, metrics,done,policy, search_val):
        metrics['obs'].append(copy.deepcopy(obs))
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['done'].append(done)
        metrics['policy'].append(policy)
        metrics['search_val_logs'].append(search_val)

    def calculate_V_from_rewards(self,metrics):
        #### We're going to use fully discounted future returns all the way to end of the episode and we return what is ultimately the value of getting to a new state.
        rewards_list = metrics['reward'].copy()
        for r in reversed(range(len(rewards_list)-1)):
            rewards_list[r] += rewards_list[r+1] * self.gamma #THIS GETS VALUE
        metrics['V'] = rewards_list
    
    