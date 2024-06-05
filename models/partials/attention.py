import torch
import torch.nn as nn
import torch.nn.functional as F

# same as original
# this should be a dot product attention
class Attention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c_s = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, query, keys, encoded_input, padding_mask):
        # query: decoder state. [N, 1, H]
        # keys: encoder states. [N, L, H]
        query = self.W_query(query)
        # attention to calculate similarity between the query and each key
        # the query is broadcast over the sequence dimension (L)
        # scores: [N, L, 1]
        scores = torch.matmul(keys, query.transpose(1, 2))
        # set padded values' attention weights to negative infinity BEFORE softmax
        scores -= torch.where(padding_mask, 1e10, 0.).unsqueeze(-1)

        # softmax to get a probability distribution over the L encoder states
        weights = F.softmax(scores, dim=-2)

        # weights: [N, L, 1]
        # encoded_input: [N, L, E] -> [N, L, H]
        # keys: [N, L, H]
        # values = keys + encoded_input (residual connection). [N, L, H]
        # first weight each value vector by broadcasting weights to each hidden dim. results in [N, L, H]
        values = self.W_c_s(encoded_input) + self.W_key(keys)
        weighted_states = weights * values
        # get a linear combination (weighted sum) of the value vectors. results in [N, H]
        weighted_states = weighted_states.sum(dim=-2) # [N, H]

        return weighted_states