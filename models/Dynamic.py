import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
from global_settings import actions_size, device, encodings_per_action, state_channels,first_layer_dyn_params, res_block_dyns
from models.res_block import resBlock
class Dynamic(torch.nn.Module):
    """
    Input: 
      state
    Notes:
     
    Outputs: 
      state

    """
    def __init__(self):
        self.action_size = actions_size
        self.first_layer_dyn_params = first_layer_dyn_params
        self.encodings_per_action = encodings_per_action
        
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = state_channels + actions_size, out_channels = state_channels, 
                                                                    kernel_size= self.first_layer_dyn_params['kernel_size'],
                                                                    stride = self.first_layer_dyn_params['stride'],
                                                                    padding = self.first_layer_dyn_params['padding'])
        self.bn1 = torch.nn.BatchNorm2d(state_channels)
        self.conv2 = torch.nn.Conv2d(in_channels = state_channels, out_channels = state_channels, 
                                                                    kernel_size= 1,
                                                                    stride = 1,
                                                                    padding = 0)
        self.bn2 = torch.nn.BatchNorm2d(state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x) for x in res_block_dyns])
        self.relu = torch.nn.ReLU()

    def forward(self,state,action):
        """
        Note on orig. shapes: 
        - state is [-1, 8, 4, 4]
        - action looks like this 1, or [[1],[2],[3]..]
        We start by creating a m x 4 x 4 x 4, where for each m, 1 of the four channels (dim 1) is all 1s and then append this.
        """

        action_plane = torch.zeros(state.shape[0],self.action_size, state.shape[2], state.shape[3]).to(device)
        action_one_hot = TF.one_hot(torch.tensor(action).to(torch.int64),actions_size).reshape(-1,self.action_size, 1, 1).to(device)
        action_plane += action_one_hot
        action_plane = action_plane.to(device)
        
        x = torch.cat((state,action_plane),dim=1)
        ### so now we have a [m,12,4,4]
        x  = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        for block in self.resBlocks:
          x = block(x)
        
        return x