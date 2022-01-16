import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 


class JakeZero(torch.nn.Module):
    
    def __init__(self, representation, dynamic, prediction):
        super().__init__()
        self.representation_network = representation
        self.dynamic_network = dynamic
        self.prediction_network = prediction

    def representation(self, x):
        return self.representation_network(x)

    def dynamic(self, state, action):
        return self.dynamic_network(state, action)
    
    def prediction(self, x):
        return self.prediction_network(x)
    