from __future__ import annotations
from typing import Annotated
from typing import TYPE_CHECKING
import random
import torch
from torch import Tensor
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from prelude import *
import torch.nn.functional as F

def sigmoid_rampup(current, rampup_length) -> float:
    # current is current epoch if doing epoch-based rampup
    """Exponential rampup from https://arxiv.org/abs/1610.02242, code from athiwaratkunThereAreMany2019"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


mse_loss = torch.nn.MSELoss(reduction='sum')
def mse_consistency_loss(
    logits_aug1: Tensor, # (N, L, C)
    logits_aug2: Tensor  # (N, L, C)
):
    N = logits_aug1.size()[0]
    V = logits_aug1.size()[2]
    return mse_loss(F.softmax(logits_aug1, dim=2), F.softmax(logits_aug2, dim=2)) / (V * N)

kl_div_loss = torch.nn.KLDivLoss(reduction='sum')
def kl_consistency_loss(
    logits_aug1: Tensor, # (N, L, C)
    logits_aug2: Tensor  # (N, L, C)
):
    N = logits_aug1.size()[0]
    V = logits_aug1.size()[2]
    return kl_div_loss(F.log_softmax(logits_aug1, dim=2), F.softmax(logits_aug2, dim=2)) / N


def augment(d_cat_tkns: Tensor, d_cat_langs: Tensor, d_cat_lens: Tensor, d_indv_lens: Tensor, transformer_d2p_d_cat_style: bool = False):
    N = d_cat_tkns.shape[0]
    d_cat_tkns_augs_list, d_cat_langs_augs_list, d_cat_lens_augs_list, d_indv_lens_augs_list = [], [], [], []
    
    for i in range(N):
        d_cat_tkn, d_cat_lang, d_cat_len, d_indv_len = d_cat_tkns[i], d_cat_langs[i], d_cat_lens[i], d_indv_lens[i]

        if not transformer_d2p_d_cat_style:
            # shuffle slices
            beginning = (d_cat_tkn[0:2], d_cat_lang[0:2], 2)
            slices = []
            slicing_head = 2
            for d_len in d_indv_len:
                slices.append((d_cat_tkn[slicing_head:slicing_head+d_len], d_cat_lang[slicing_head:slicing_head+d_len], d_len))
                slicing_head += d_len
            ending = (d_cat_tkn[slicing_head:slicing_head+1], d_cat_lang[slicing_head:slicing_head+1], 1)
            random.shuffle(slices)
            
            # drop one slice at 50% chance
            if len(slices) > 1 and random.random() < 0.5:
                slices.pop()
                
            # reassemble
            d_cat_tkn_auged = torch.cat([beginning[0]] + [slice[0] for slice in slices] + [ending[0]])
            d_cat_lang_auged = torch.cat([beginning[1]] + [slice[1] for slice in slices] + [ending[1]])
            d_indv_len_auged = torch.tensor([slice[2] for slice in slices])
            d_cat_len_auged = torch.tensor([2 + sum([slice[2] for slice in slices]) + 1])
            
        else:
            
            slices = []
            slicing_head = 0
            for d_len in d_indv_len:
                slices.append((d_cat_tkn[slicing_head:slicing_head+d_len], d_cat_lang[slicing_head:slicing_head+d_len], d_len))
                slicing_head += d_len
            random.shuffle(slices)

            # drop one slice at 50% chance
            if len(slices) > 1 and random.random() < 0.5:
                slices.pop()
                
            # reassemble
            d_cat_tkn_auged = torch.cat([slice[0] for slice in slices])
            d_cat_lang_auged = torch.cat([slice[1] for slice in slices])
            d_indv_len_auged = torch.tensor([slice[2] for slice in slices])
            d_cat_len_auged = torch.tensor([sum([slice[2] for slice in slices])])

        d_cat_tkns_augs_list.append(d_cat_tkn_auged)
        d_cat_langs_augs_list.append(d_cat_lang_auged)
        d_cat_lens_augs_list.append(d_cat_len_auged)
        d_indv_lens_augs_list.append(d_indv_len_auged)


    # .get_device() not working for cpu?
    d_cat_tkns_a = pad_sequence(d_cat_tkns_augs_list, batch_first=True, padding_value=PAD_IDX).to(d_cat_tkns.device)
    d_cat_langs_a = pad_sequence(d_cat_langs_augs_list, batch_first=True, padding_value=PAD_IDX).to(d_cat_tkns.device)
    d_cat_lens_a = torch.stack(d_cat_lens_augs_list, dim=0).to(d_cat_tkns.device)
    d_indv_lens_a = pad_sequence(d_indv_lens_augs_list, batch_first=True, padding_value=0).to(d_cat_tkns.device)

    return d_cat_tkns_a, d_cat_langs_a, d_cat_lens_a, d_indv_lens_a    
