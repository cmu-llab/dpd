import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP to predict probs from decoder state
class MLP(nn.Module):
    def __init__(self, input_dim, feedforward_dim, output_size):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2 * feedforward_dim),
            nn.ReLU(),
            nn.Linear(2 * feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, output_size, bias=False)
        )
        
    def forward(self, decoder_state):
        scores = self.network(decoder_state)        
        return scores # need to do softmax / log softmax manually depending on context