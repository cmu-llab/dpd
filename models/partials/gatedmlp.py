import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange

# Gated MLP is like an MLP but part of the model is enabled based on some discrete input, such as target language.
class GatedMLP(nn.Module):
    def __init__(self, input_dim, feedforward_dim, gate_count, output_size):
        super().__init__()
        
        self.l1 = nn.Linear(input_dim, 2 * (1 + gate_count) * feedforward_dim)
        self.l2 = nn.Linear(2 * feedforward_dim, feedforward_dim)
        self.l3 = nn.Linear(feedforward_dim, output_size, bias=False)
        
        self.G = gate_count
        self.D = feedforward_dim
        self.gate_scale = 0.9
        
    def forward(self, 
        classification_input: Tensor, # (N, input_dim)
        gate_ids: Tensor, # (N, 1)
    ):
        assert torch.max(gate_ids) < self.G
        N = classification_input.shape[0] # batch size
        
        out = self.l1(classification_input) # (N, 2 * (1 + G) * D)
        out = F.relu(out)
        
        mask_ = F.one_hot(gate_ids, num_classes = self.G) # (N, 1, G), .
        mask_ = rearrange(mask_, 'N 1 G -> N G') # (N, G)
        scaled_mask_ = mask_ * self.gate_scale # (N, G)
        
        scaled_mask_ = torch.cat((scaled_mask_, torch.ones(N, 1, device = classification_input.device)), dim = 1) # (N, 1 + G) last gate always pass everything
        scaled_mask = repeat(scaled_mask_, 'N GPlusOne -> N GPlusOne D', D = 2 * self.D)
        mask = rearrange(scaled_mask, 'N GPlusOne D -> N (GPlusOne D)') # (N, 2 * (1 + gate_count) * D)
        
        out = out * mask # (N, 2 * (1 + gate_count) * D)
        out = rearrange(out, 'N (col row) -> N col row', col = (1 + self.G), row = 2 * self.D) # (N, 1 + gate_count, 2 * D)
        out = torch.sum(out, dim = 1) # (N, 2 * D)
        
        out = self.l2(out) # (N, D)
        out = F.relu(out)
        
        scores = self.l3(out) # (N, output_size)
        return scores
        # need to do softmax / log softmax manually depending on context
        
class GatedMLPConcat(nn.Module):
    def __init__(self, input_dim, feedforward_dim, gate_count, output_size):
        super().__init__()
        
        SM = 2 # straight multiplier
        GM = 1    # gated multiplier
        
        self.l1_straight = nn.Linear(input_dim, SM * feedforward_dim) # straight through for any target
        self.l1_gated = nn.Linear(input_dim, gate_count * (GM * feedforward_dim)) # gated for each target
        self.l2 = nn.Linear((SM + GM) * feedforward_dim, feedforward_dim)
        self.l3 = nn.Linear(feedforward_dim, output_size, bias=False)
        
        self.G = gate_count
        self.D = feedforward_dim
        self.SM = SM
        self.GM = GM
        
    def forward(self, 
        classification_input: Tensor, # (N, input_dim)
        gate_ids: Tensor, # (N, 1)
    ):
        assert torch.max(gate_ids) < self.G
        N = classification_input.shape[0] # batch size
        
        out_straight = self.l1_straight(classification_input) # (N, SM * D)
        out_straight = F.relu(out_straight)
        
        out_gated = self.l1_gated(classification_input) # (N, G * GM * D)
        out_gated = F.relu(out_gated)
        
        mask_ = F.one_hot(gate_ids, num_classes = self.G) # (N, 1, G), .
        mask_ = rearrange(mask_, 'N 1 G -> N G') # (N, G)
        
        mask_ = repeat(mask_, 'N G -> N G D', D = self.GM * self.D)
        mask = rearrange(mask_, 'N G D -> N (G D)') # (N, GM * G * D)
        
        out_gated = out_gated * mask # (N, GM * G * D)
        out_gated = rearrange(out_gated, 'N (col row) -> N col row', col = self.G, row = self.GM * self.D) # (N, G, GM * D)
        out_gated = torch.sum(out_gated, dim = 1) # (N, GM * D)
        
        out_concat = torch.cat((out_straight, out_gated), dim = 1) # (N, (SM + GM) * D) 
        
        out = self.l2(out_concat) # (N, D)
        out = F.relu(out)
        
        scores = self.l3(out) # (N, output_size)
        return scores
        # need to do softmax / log softmax manually depending on context
        
