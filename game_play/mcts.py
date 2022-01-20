import time
import numpy as np
from global_settings import value_only, manual_over_ride_play_limit, temperature_changes, play_limit_mcts, c1, c2 , gamma, actions_size, play_limit_mcts, exponent_node_n, use_policy, dirichlet_alpha, ucb_denom_k, reward_support, value_support
from utils import support_to_scalar
import torch

class Node:
    def __init__(self,parent, p = 0):
        self.Q_sum = 0
        self.Q = 0
        self.N = 0
        self.P = p
        self.parent = parent
        self.children = []
        self.r = 0
        self.evaluated = False
        self.move = 0

class MCTS:
    def __init__(self, episode, epoch = 1, pick_best=False):
        #during testing we use pick_best = True, so temperatuer is irrelevant
        self.epoch = epoch
        self.set_temperature_and_sims()
        self.pick_best = pick_best
        self.ep = episode
        self.c1, self.c2, self.gamma = c1,c2, gamma
        self.num_actions = actions_size
        self.dir_alpha = dirichlet_alpha        

    def set_temperature_and_sims(self):
        for key, val in temperature_changes.items():
            if self.epoch > key:
                self.temperature = val
        if manual_over_ride_play_limit == None:
            for key, val in play_limit_mcts.items():
                if self.epoch > key:
                    self.sims = val
        else:
             self.sims = manual_over_ride_play_limit

    def one_turn(self,root_node):
        
        self.root_node = root_node         
        idx = 0 
        for _ in range(self.sims):
            idx+=1
            self.mcts_go(root_node)
        if self.pick_best:
            policy, chosen_action = self.pick_best_action(root_node)
        else:
            policy, chosen_action = self.randomly_sample_action(self.root_node)

        return policy, chosen_action, self.root_node.Q, self.root_node.V
        
    def mcts_go(self,node):
        if not node.evaluated:
            self.expand(node)
        else:
            best_ucb_child = self.pick_child(node)
            self.mcts_go(best_ucb_child)
                
    def expand(self,node):
        """You've reached an unevaluated node
        First evalute it by calculating r and V, set evaluated flag to True
        then add unevaluated children
        """
        node.evaluated = True
        if node != self.root_node:
            
            state, r = self.ep.dyn_model(node.parent.state,node.move) #
            node.r = r
            node.state = state
            r = support_to_scalar(r[0].detach().numpy(),*reward_support)
            
            
            # print("unit test line 77 mcts: printing the number of times the dyn model is run in a single go")

        prob_action,V = self.ep.prediction_model(node.state) #this will come out as a 1 x 4 and 1x1
        prob_action, V = prob_action[0], V[0]
        
        V = support_to_scalar(V.detach().numpy(), *value_support)
        
        

        if node!= self.root_node:
            self.back_prop_rewards(node, V)
        else:
            prob_action = 0.75* prob_action
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.dir_alpha]*actions_size).float())
            sample_dir = dir.sample()
            prob_action += 0.25*sample_dir
            self.root_node.V = V
        
        ## Add a child node for each action of this node.
        for edge in range(self.num_actions):
            new_node = Node(parent=node, p=prob_action[edge])
            new_node.Q = node.Q
            new_node.move = edge
            node.children.append(new_node)
        
        
    def pick_child(self,node):
        ucbs = [self.UCB_calc(x).numpy() for x in node.children]
        # return node.children[np.argmax([self.UCB_calc(x).numpy() for x in node.children])]
        return node.children[ucbs.index(max(ucbs))]
    
    def UCB_calc(self,node):
        if use_policy:
            policy_and_novelty_coef = node.P * np.sqrt(node.parent.N) / (ucb_denom_k+node.N**exponent_node_n)
        else:
            policy_and_novelty_coef = 0*node.P + np.sqrt(1/2) * np.log(node.parent.N + 1) / (1+ node.N)
        muZeroModerator = self.c1 + np.log((node.parent.N + self.c2 + self.c1+1)/self.c2) #this is basically 1.
        return node.Q + policy_and_novelty_coef * muZeroModerator
    
    def back_prop_rewards(self, node, V):
        """just send those rewards up the chain"""
        node.Q_sum += V*gamma 
        if not value_only:
            node.Q_sum += node.r
        node.N +=1
        node.Q = node.Q_sum / node.N
        # print("mcts unit test: here are some samples of node V, r , Q: ", V, node.r,node.Q)
        if node != self.root_node:
            self.back_prop_rewards(node.parent, node.Q)

    def randomly_sample_action(self,root_node):
        policy = np.array([float(x.N) ** (1 / self.temperature) for x in root_node.children])
        
        policy = policy / np.sum(policy)
        
        return policy, np.random.choice(list(range(actions_size)), p=policy)

    def pick_best_action(self, root_node):
        policy = np.array([float(x.N) for x in root_node.children])
        policy = policy / np.sum(policy)
        
        return policy, np.argmax(policy)