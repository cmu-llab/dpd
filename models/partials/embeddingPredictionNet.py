import torch
import torch.nn as nn
import torch.nn.functional as F

# A fully connected NN aimed to predict embedding of intermediate text space from hidden state where the intermediate form is constructed
class EmbeddingPredictionNet(nn.Module):
    def __init__(self, hidden_dim, feedforward_dim, enbedding_dim):
        super(EmbeddingPredictionNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 2 * feedforward_dim),
            nn.ReLU(),
            nn.Linear(2 * feedforward_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, enbedding_dim, bias=False)
        )
        
    # this will be fed to mean error loss / cos proximity loss?
    def forward(self, decoder_state):
        embedding_hat = self.network(decoder_state)        
        return embedding_hat