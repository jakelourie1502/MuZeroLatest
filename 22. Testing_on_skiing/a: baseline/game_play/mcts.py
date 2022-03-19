from logging import root
import time
import numpy as np
from global_settings import exp_gamma, exp_v_support, mcts_update_mode, mcts_rolling_alpha, rdn_beta, value_only, manual_over_ride_play_limit
from global_settings import temperature_changes, play_limit_mcts, c1, c2 , gamma, actions_size, play_limit_mcts, exponent_node_n, use_policy, dirichlet_alpha
from global_settings import exploration_strategy, ucb_denom_k, reward_support, value_support
from utils import support_to_scalar
import torch

class Node:
    def __init__(self,parent, p = 0):
        self.Q_sum = 0
        self.Q = 0
        self.Qe_sum = 0
        self.Qe = 0
        self.total_Q = 0
        self.N = 0
        self.p = p
        self.parent = parent
        self.children = []
        self.r = 0
        self.v =0
        self.exp_r= 0
        self.exp_v = 0
        self.d = 0
        self.evaluated = False
        self.move = 0

class MCTS:
    def __init__(self, RDN_OBJ, episode, epoch = 1, pick_best=False):
        #during testing we use pick_best = True, so temperatuer is irrelevant
        self.epoch = epoch
        self.set_temperature_and_sims()
        self.pick_best = pick_best
        self.ep = episode
        self.c1, self.c2, self.gamma, self.exp_gamma = c1,c2, gamma, exp_gamma
        self.num_actions = actions_size
        self.dir_alpha = dirichlet_alpha        
        self.RDN_eval = RDN_OBJ
        self.update_mode = mcts_update_mode 
        self.rolling_alpha = mcts_rolling_alpha

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

        # ungathered_novelty = np.max([x.exp_r*(1-x.d) for x in root_node.children if x.move != chosen_action])
        immediate_novelty = root_node.children[chosen_action].exp_r
        return policy, chosen_action, self.root_node.Q, self.root_node.v, immediate_novelty, self.root_node.exp_v
        
    def mcts_go(self,node):
        if not node.evaluated:
            
            self.expand(node)
        else:
            
            best_ucb_child = self.pick_child(node)
            
            self.mcts_go(best_ucb_child)
                
    def expand(self,node):
        """You've reached an unevaluated node
        First evalute it by calculating r and V and d, set evaluated flag to True
        then add unevaluated children
        """
        node.evaluated = True
        
        if node != self.root_node:
            ## GET TERMINAL INFO
            state, r, d = self.ep.dyn_model(node.parent.state,node.move) #
            d = d[0][0].detach().numpy()
            node.d = d
            node.state = state

            ## Get r and exp_r
            ##### ONLY IF EXPLORATION ON
            if exploration_strategy != 'none':
                node.r = support_to_scalar(r[0].detach().numpy(),*reward_support)
                rdn_random = self.ep.rdn(state)
                rdn_pred = self.ep.rdn_pred(state)
                if node.parent == self.root_node: #only using lead nodes for logging mean of RDN
                    log = True
                else:
                    log = False
                rdn_value = self.RDN_eval.evaluate(rdn_random,rdn_pred, log)
                node.exp_r = (1-node.parent.d)*rdn_value.detach().numpy()
        
        ## Get V and exp_V
        prob_action, v, exp_v = self.ep.prediction_model(node.state)
        prob_action, v, exp_v = prob_action[0], v[0], exp_v[0]
        v = support_to_scalar(v.detach().numpy(), *value_support)
        node.v = v

        ### ONLY IF EXPLORATION - otherwise stays set to 0.
        if exploration_strategy == "full":
            exp_v = support_to_scalar(exp_v.detach().numpy(),*exp_v_support)
            node.exp_v = (1-node.d)*exp_v
        else:
            exp_v = 0
        if node!= self.root_node:
            self.back_prop_rewards(node, v/gamma, exp_v/exp_gamma) ##divide by gamma because we want to nullify the first discounting in backprop
        else:
            prob_action = 0.5* prob_action
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.dir_alpha]*actions_size).float())
            sample_dir = dir.sample()
            prob_action += 0.5*sample_dir
            self.root_node.N +=1
            
        ## Add a child node for each action of this node.
        for edge in range(self.num_actions):
            
            new_node = Node(parent=node, p=prob_action[edge])
            new_node.total_Q = node.total_Q
            new_node.move = edge
            node.children.append(new_node)
        
        
    def pick_child(self,node):
        ucbs = [self.UCB_calc(x).numpy() for x in node.children]
    
        # return node.children[np.argmax([self.UCB_calc(x).numpy() for x in node.children])]
        return node.children[ucbs.index(max(ucbs))]
    
    def UCB_calc(self,node):
        
        if use_policy:
            policy_and_novelty_coef = node.p* np.sqrt(node.parent.N) / (ucb_denom_k+node.N**exponent_node_n)
        else:
            policy_and_novelty_coef = 0*node.p + np.sqrt(1/2) * np.log(node.parent.N + 1) / (1+ node.N)
        muZeroModerator = self.c1 + np.log((node.parent.N + self.c2 + self.c1+1)/self.c2) #this is basically 1.
        return node.total_Q + policy_and_novelty_coef * muZeroModerator
    
    def back_prop_rewards(self, node, v, exp_v):
        """
        For real reward, we want to weight V by the future state we never reach with the current prediction for value
        """
        exp_v = exp_v*self.exp_gamma
        exp_v += node.exp_r #### ONLY IF exploration on.
        v = v*gamma
        v = (1-node.d)*v + node.d*node.v
        if not value_only:
            v+= node.r
        node.N +=1

        if self.update_mode == 'mean':
            node.Qe_sum += rdn_beta *exp_v
            node.Q_sum += v
            node.Qe = node.Qe_sum / node.N
            node.Q = node.Q_sum / node.N

        ###With Max, we work out whether the overall Q value can be increased. If so, we update both novelty and intrinsic to the values just received

        elif self.update_mode == 'max':
            candidate_Q = max(0,rdn_beta*exp_v) + v
            if candidate_Q > node.total_Q:
                node.Q = v
                node.Qe = rdn_beta*exp_v

        #### With rolling; we walso want to work out whether we have reached a new max, in which case we follow max
        #Otherwise, we update the rolling average of novelty and intrinsic reward with values just received.

        elif self.update_mode == 'rolling':
            candidate_Q = max(0,rdn_beta*exp_v) + v
            if candidate_Q > node.total_Q:
                node.Q = v
                node.Qe = rdn_beta*exp_v
            else: 
                node.Q += self.rolling_alpha*(v - node.Q)
                node.Qe += self.rolling_alpha*(rdn_beta*exp_v - node.Qe)
        
        if self.pick_best: 
            node.total_Q = node.Q
        else:
            node.total_Q = max(0,node.Qe) + node.Q
            
        if np.random.uniform(0,1000000) < 2:
            print("Checking the total Q and that Qe samples", node.total_Q, node.Qe)
        if node != self.root_node:
            self.back_prop_rewards(node.parent, v, exp_v)

    def randomly_sample_action(self,root_node):
        policy = np.array([float(x.N) ** (1 / self.temperature) for x in root_node.children])
        # print([float(x.Q) for x in root_node.children])
        policy = policy / np.sum(policy)
        
        return policy, np.random.choice(list(range(actions_size)), p=policy)

    def pick_best_action(self, root_node):
        policy = np.array([float(x.N) for x in root_node.children])
        policy = policy / np.sum(policy)
        
        return policy, np.argmax(policy)