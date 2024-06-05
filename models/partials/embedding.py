import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from specialtokens import *

class Embedding(nn.Module):
    def __init__(self, embedding_dim, num_ipa_tokens, num_langs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_ipa_tokens = num_ipa_tokens
        self.num_langs = num_langs        
        
        self.char_embeddings = nn.Embedding(self.num_ipa_tokens, self.embedding_dim, padding_idx=PAD_IDX)
        self.lang_embeddings = nn.Embedding(self.num_langs, self.embedding_dim, padding_idx=PAD_IDX)
        
        self.fc = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

    def set_pad_embedding_to_zero(self):
        self.char_embeddings.weight.data[PAD_IDX].fill_(0)
        self.lang_embeddings.weight.data[PAD_IDX].fill_(0)

    # both args are size (N, seq len). The second one is optional. Add lang_ids if wish to slap on some lang embedding
    def forward(self, char_ids: Tensor, lang_ids: Tensor | None = None) -> Tensor:
        
        chars_embedded = self.char_embeddings(char_ids) # (N, seq, embedding size)
        if lang_ids == None:
            return chars_embedded
        else:
            lang_embedded = self.lang_embeddings(lang_ids) # (N, seq, embedding size)
            embedding_cat = torch.cat((chars_embedded, lang_embedded), dim=-1) # (N, seq, 2 * embedding size)
            combined_embedding = self.fc(embedding_cat) # (N, seq, embedding size)
            return combined_embedding
