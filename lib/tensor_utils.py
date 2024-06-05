import torch
from torch import Tensor
from specialtokens import *
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, unpack_sequence

def random_mask_by_proportion(
    length: int,
    proportion: float,
):
    num_ones = round(length * proportion)
    num_zeros = length - num_ones
    mask = torch.cat((torch.zeros(num_zeros), torch.ones(num_ones)))
    mask = mask[torch.randperm(mask.size(0))]
    return mask


# generate a random mask of given length with a proportion, but in a way that lower porportion masks will be a subset of highr porportion masks
def random_mask_by_proportion_subsets(
    length: int,
    proportion: float,
):
    x = torch.rand(length)
    y = torch.sort(x)
    num_ones = round(length * proportion)
    upperbound = -1 if num_ones == 0 else y.values[num_ones - 1]
    mask_bool = x <= upperbound
    mask = mask_bool.float()
    return mask



def sequences_equal(
    seqs1: Tensor, # (N, L1), padded sequences
    seqs2: Tensor, # (N, L2), padded sequences
) -> Tensor: # (N)
    L1 = seqs1.shape[1]
    L2 = seqs2.shape[1]
    if L1 < L2:
        seqs1 = F.pad(seqs1, (0, L2 - L1), value=PAD_IDX)
    if L2 < L1:
        seqs2 = F.pad(seqs2, (0, L1 - L2), value=PAD_IDX)
    return torch.all(seqs1 == seqs2, dim=-1)

def num_sequences_equal(
    seqs1: Tensor, # (N, L1), padded sequences
    seqs2: Tensor, # (N, L2), padded sequences
) -> int:
    return torch.sum(sequences_equal(seqs1, seqs2)).item()

        
# https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/utils.html#padded_stack, MIT license
def padded_stack(
    tensors: list[torch.Tensor], side: str = "right", mode: str = "constant", value: int | float = 0
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out


# based on https://discuss.pytorch.org/t/how-to-sort-tensor-by-given-order/61625
def sort_by_permutation_3d(x, permutation):
    d1, d2, d3 = x.size()
    return x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2, d3)


def collate_endnodes_to_tensors(N, Nc, Nd, processed_endnodes, device):

    # collapse processed_endnodes into a single tensor
    collated_candidate_list = []
    collated_normalized_log_prob_list = []
    collated_log_prob_list = []
    for endnodes_for_batch in processed_endnodes:
        candidate_list_for_batch = []
        normalized_log_prob_list_for_batch = []
        log_prob_list_for_batch = []
        for i, endnode in enumerate(endnodes_for_batch): 
            if i >= Nc:
                break
            candidate_list_for_batch.append(endnode['endnode_seq'])
            normalized_log_prob_list_for_batch.append(endnode['normalized_log_prob'])
            log_prob_list_for_batch.append(endnode['log_prob'])
        while len(candidate_list_for_batch) < Nc: # pad to Nc if less
            candidate_list_for_batch.append(torch.LongTensor([PAD_IDX]).to(device))
            normalized_log_prob_list_for_batch.append(torch.Tensor([[-float('inf')]]).to(device))
            log_prob_list_for_batch.append(torch.Tensor([[-float('inf')]]))
        collated_candidate_list.append(pad_sequence(candidate_list_for_batch, batch_first=True, padding_value=PAD_IDX))
        collated_log_prob_list.append(torch.tensor(log_prob_list_for_batch).to(device))
        collated_normalized_log_prob_list.append(torch.tensor(normalized_log_prob_list_for_batch).to(device))
    # collated_candidate_list: list[N] of (Nc, Lc)
    # collated_normalized_log_prob_list: list[N] of (Nc)
    # collated_log_prob_list: list[N] of (Nc)

    collated_candidates = padded_stack(collated_candidate_list, mode='constant', value=PAD_IDX) # (N, Nc, Lc)
    
    collated_normalized_log_prob = repeat(torch.stack(collated_normalized_log_prob_list, dim=0), 'N Nc -> N Nc 1') # (N, Nc, 1)
    collated_log_prob = repeat(torch.stack(collated_log_prob_list, dim=0), 'N Nc -> N Nc 1') # (N, Nc, 1)
    # collated_normalized_log_prob = repeat(collated_normalized_log_prob_, 'N Nc 1 -> N Nc Nd 1', Nd=Nd) # (N, Nc, Nd, 1)
    # collated_log_prob = repeat(collated_log_prob_, 'N Nc 1 -> N Nc Nd 1', Nd=Nd) # (N, Nc, Nd, 1)
    
    return collated_candidates, collated_normalized_log_prob, collated_log_prob
