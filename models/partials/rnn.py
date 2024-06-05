import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GRURNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional: bool = False):
        super(GRURNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        match h0:
            case None:            
                return self.rnn(x)
            case _:
                return self.rnn(x, h0)
