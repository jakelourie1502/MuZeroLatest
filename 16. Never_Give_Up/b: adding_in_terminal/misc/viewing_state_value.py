import time
from pickle import FALSE
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from math import factorial as F
import matplotlib.pyplot as plt
from utils import support_to_scalar
import numpy as np
import torch
# from training_and_data.training import loss_func_RDN


from global_settings import lakes, goals, env_size, lake_coverage, actions_size, max_plays, dist, play_random, value_support
from game_play.frozen_lake import gridWorld
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
jake_zero = torch.load('../saved_models/jake_zero',map_location=torch.device('cpu'))
pred = jake_zero.prediction
dyn = jake_zero.dynamic
repr =jake_zero.representation
jake_zero.eval()
#1
dist = np.zeros((4*12+1))

def RDN_score(starting_pos,action):
    dist = np.zeros((4*12+1))
    dist[starting_pos] = 1
    env=gridWorld(env_size,lakes, goals, n_actions = actions_size, max_steps = 8, dist = dist, seed = None, rnd=play_random,lake_cov=lake_coverage, randomly_create_env = False)
    raw_obs = env.reset()
    # print(raw_obs)
    obs = vector2d_to_tensor_for_model(torch.tensor(raw_obs).float())
    s = jake_zero.representation(obs)
    p, v = jake_zero.prediction(s)
    print("policy: ", p)
    
    RDN_random_output = jake_zero.RDN(s).detach()
    RDN_prediction = jake_zero.RDN_prediction(s).detach()
    loss_RDN = jake_zero.rdn_obj.evaluate(RDN_random_output,RDN_prediction)
    print("RDN in starting place: ", loss_RDN)
    s, _, d = jake_zero.dynamic(s,action)
    # s, _, d = jake_zero.dynamic(s,1)
    
    # s, _, d = jake_zero.dynamic(s,1)
    print("terminal possibility: ",d)
    
    RDN_random_output = jake_zero.RDN(s).detach()
    RDN_prediction = jake_zero.RDN_prediction(s).detach()
    loss_RDN = jake_zero.rdn_obj.evaluate(RDN_random_output,RDN_prediction)
    print("RDN in after moving right place: ", loss_RDN)

s_p =21
RDN_score(s_p,0)
RDN_score(s_p,1)
RDN_score(s_p,2)
RDN_score(s_p,3)


models = [repr, dyn, pred]
epsilon = 0
temperature = 0.01
ep_h = []
def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero,epsilon)
        metrics, rew = ep.play_episode(jake_zero.rdn_obj,False, epoch = 1000, view_game =True)
        # replay_buffer.add_ep_log(metrics)
        print(rew)
        ep_h.append(rew)


def plot_beta(a,b):

    def normalisation_beta(a,b):
        return 1/ (F(a-1)*F(b-1) / F(a+b-1))

    def beta_val(a,b,val):
        return normalisation_beta(a,b) * val**(a-1) * (1-val)**(b-1)
    
    X = np.linspace(0,1,100)
    y = [beta_val(a,b,x) for x in X]
    plt.plot(X,y)

tn = time.time()
fig = plt.figure()
plt.savefig('saved_plt.png')

for _ in range(2):
    Play_Episode_Wrapper()
    a = 1 + np.sum(ep_h)
    b = 2 + len(ep_h) - np.sum(ep_h)
    plot_beta(a,b)
    plt.savefig('saved_plt.png')

    plt.pause(0.001)
    plt.close()


# print(np.sum(ep_h), np.mean(ep_h))
print(time.time()- tn)