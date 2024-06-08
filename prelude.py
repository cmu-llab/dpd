import torch
from einops import rearrange, repeat
import pytorch_lightning as pl
from specialtokens import *
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
import itertools
import os
import json
from torch import Tensor

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


rescored_endnodes_t = tuple[Tensor, Tensor, Tensor, Tensor, Tensor] 
batch_t = tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
ALL_TARGET_LANGS_LABEL: str = 'all_target_langs'

from enum import Enum

# strategy identifiers
class STRATS(Enum):
    Supv = "Supv"
    SupvBst = "SupvBst"
    Pi = "Pi"
    PiBst = "PiBst"
    Bpall = "Bpall"
    BpallBst = "BpallBst"
    BpallPi = "BpallPi"
    BpallPiBst = "BpallPiBst"

class DATASETS(Enum):
    NRO_IPA = "Nromance_ipa"
    WIKIHAN = "chinese_wikihan2022"
    
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LabelStatus(Enum):
    UNLABELLED = 0
    PSEUDOLABEL = 1
    LABELLED = 2
    REVEALED = 3 # was hidden but revealed for evaluation
