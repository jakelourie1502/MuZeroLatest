from pickle import FALSE
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch



from global_settings import lakes, goals, env_size, lake_coverage, actions_size, max_plays, dist, play_random
from game_play.frozen_lake import gridWorld
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
jake_zero = torch.load('../saved_models/jake_zero',map_location=torch.device('cpu'))
pred = jake_zero.prediction
dyn = jake_zero.dynamic
repr =jake_zero.representation
jake_zero.eval()
#1
dist = np.zeros((37))
dist[4] = 1

env=gridWorld(env_size,[(2, 2), (5, 1)], {(0, 5): 1}, n_actions = actions_size, max_steps = 8, dist = dist, seed = None, rnd=play_random,lake_cov=lake_coverage, randomly_create_env = True)
# env.generate_random_lakes(np.random.uniform(lake_coverage[0],lake_coverage[1]))
obs = env.reset()
print(obs)
print('values for initial obs')
move = 1
print("initial policy:")
print(pred(repr(vector2d_to_tensor_for_model(obs).float()))[0])
print("initial value of state")
print(pred(repr(vector2d_to_tensor_for_model(obs).float()))[1])
for i in range(4):
    obs = env.reset()
    # obs,_ , _, _ = env.step(3)
    obs = vector2d_to_tensor_for_model(obs)
    print(pred(dyn(repr(obs.float()),i)))

'Values after moving in a direction'
for i in range(4):
    obs = env.reset()
    obs, _, reward, done = env.step(move)
    obs, _, reward, done = env.step(move)
    obs, _, reward, done = env.step(move)
    
    if i == 0: 
        print(obs)
        print(done)
    obs = vector2d_to_tensor_for_model(obs)
    print(pred(dyn(repr(obs.float()),i)))

print("Same but state transitions" )
for i in range(4):
    obs = env.reset()
    s = dyn(repr(vector2d_to_tensor_for_model(obs).float()),move)
    s = dyn(s,move)
    s = dyn(s,move)
    s = dyn(s,i)
    print(pred(s))

# 'Values after moving in direcion three times'
# for i in range(4):
#     obs = env.reset()
#     obs, _, reward, done = env.step(move)
#     obs, _, reward, done = env.step(3)
#     obs, _, reward, done = env.step(3)
#     obs, _, reward, done = env.step(0)
#     if i == 0: 
#         print(obs)
#         print(done)
#     obs = vector2d_to_tensor_for_model(obs)
#     print(pred(dyn(repr(obs.float()),i)))

# print("Same but state transitions" )
# for i in range(4):
#     obs = env.reset()
#     s = dyn(repr(vector2d_to_tensor_for_model(obs).float()),move)
#     s = dyn(s,3)
#     s = dyn(s,3)
#     s = dyn(s,0)
#     s = dyn(s,i)
#     print(pred(s))
print("Moving twice in same direction through water to goal")
move = 0
obs = env.reset()
s = dyn(repr(vector2d_to_tensor_for_model(obs).float()),move)
s = dyn(s, move)
s = dyn(s, 0)
print(pred(s)[1])

models = [repr, dyn, pred]
epsilon = 0
temperature = 0.01
ep_h = []
def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero,epsilon)
        metrics, rew = ep.play_episode(True, epoch = 10000, view_game =False)
        # replay_buffer.add_ep_log(metrics)
        print(rew)
        ep_h.append(rew)
import time
tn = time.time()
for _ in range(100):
    Play_Episode_Wrapper()
print(np.sum(ep_h), np.mean(ep_h))
print(time.time()- tn)