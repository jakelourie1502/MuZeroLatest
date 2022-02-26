import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
from global_settings import env_size, state_channels,value_conv_channels,policy_conv_channels,res_block_pred, policy_output_supp, policy_hidden_dim, value_hidden_dim, value_support
from models.res_block import resBlock
class Prediction(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self):
        super().__init__()
        
        
        self.resBlocks = torch.nn.ModuleList([resBlock(x) for x in res_block_pred])
        #value
        self.value_conv = torch.nn.Conv2d(state_channels,value_conv_channels,1,1,0)
        self.bn1v = torch.nn.BatchNorm2d(value_conv_channels)
        self.FChidV = torch.nn.Linear(env_size[0]*env_size[1]*value_conv_channels, value_hidden_dim)
        self.bn2v = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutV = torch.nn.Linear(value_hidden_dim, value_support[2])
        
        #policy
        self.policy_conv = torch.nn.Conv2d(state_channels,policy_conv_channels,1,1,0)
        self.bn1p = torch.nn.BatchNorm2d(policy_conv_channels)
        self.FChidp = torch.nn.Linear(env_size[0]*env_size[1]*policy_conv_channels, policy_hidden_dim)
        self.bn2p = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutP = torch.nn.Linear(policy_hidden_dim, policy_output_supp)
        
        #activation
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.sm_v = torch.nn.Softmax(dim=1)
        self.sm_p = torch.nn.Softmax(dim=1)

    def forward(self,state):
        # x = torch.flatten(state,start_dim=1)
        for block in self.resBlocks:
          state = block(state)
        
        ##value
        v = state 
        v = self.value_conv(v)
        v = self.bn1v(v)
        v = self.relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.FChidV(v)
        v = self.bn2v(v)
        v = self.relu(v)
        v = self.FCoutV(v)
        v = self.sm_v(v)

        ##policy
        p = state
        p = self.policy_conv(p)
        p = self.bn1p(p)
        p = self.relu(p)
        p = torch.flatten(p, start_dim=1)
        p = self.FChidp(p)
        p = self.bn2p(p)
        p = self.relu(p)
        p = self.FCoutP(p)
        p = self.sm_p(p)
        return p,v