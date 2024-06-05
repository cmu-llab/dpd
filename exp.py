# region === imports ===
import torch
import pytorch_lightning as pl
# from lib.analysis_utils import *
import wandb
from lib.dataloader_manager import DataloaderManager
from models.encoderDecoderRNN import Seq2SeqRNN
import models.biDirReconStrategies
from specialtokens import *
import os
from pprint import pprint
import json
import sys
import torch.nn as nn
from models.biDirReconIntegration import biDirReconModelRNN, biDirReconModelTrans
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from models.semisupervised_strats.bootstrapping import Pseudolabeller
import logging
import random
logging.getLogger('lingpy').setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_printoptions(precision=4)
torch.set_printoptions(sci_mode=False)
import warnings
warnings.filterwarnings(action="ignore", message=".*num_workers.*")
warnings.filterwarnings(action="ignore", message=".*negatively affect performance.*")
warnings.filterwarnings(action="ignore", message=".*MPS available.*")
import lib.getfreegpu
from pytorch_lightning.callbacks.callback import Callback
from dotenv import load_dotenv
import os
import argparse
from prelude import STRATS, str2bool
# endregion

load_dotenv()
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
print("== Environment ==") 
pprint({
    "WANDB_ENTITY": WANDB_ENTITY, 
    "WANDB_PROJECT": WANDB_PROJECT
})
def get_strat_id(strat: str, bootstrapping: bool) -> STRATS:
    match (strat, bootstrapping):
        case ("supervised_only", False):
            return STRATS.Supv
        case ("supervised_only", True):
            return STRATS.SupvBst
        case ("pimodel", False):
            return STRATS.Pi
        case ("pimodel", True):
            return STRATS.PiBst
        case ("bpall_cringe", False):
            return STRATS.Bpall
        case ("bpall_cringe", True):
            return STRATS.BpallBst
        case ("pimodel_bpall_cringe", False):
            return STRATS.BpallPi
        case ("pimodel_bpall_cringe", True):
            return STRATS.BpallPiBst

def get_strat_archi_id(strat: str, bootstrapping: bool, architecture: str) -> str:
    strat_id = get_strat_id(strat, bootstrapping)
    match architecture:
        case 'GRU':
            return f"GRU{strat_id.value}"
        case 'Transformer':
            return f"Trans{strat_id.value}"
            

# region === arg parsing ===

parser = argparse.ArgumentParser(description='Run experiment')

# script
parser.add_argument('--countparam', action='store_true', default=False, help='print param count and exit')
parser.add_argument('--dev', action='store_true', default=False)
parser.add_argument('--nowandb', action='store_true', default=False)
parser.add_argument('--noprogressbar', action='store_true', default=False)
parser.add_argument('--wandbwatch', action='store_true', default=False)
parser.add_argument('--logmodel', action='store_true', default=False)
parser.add_argument('--sweeping', action='store_true', default=False)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--gpu', type=int, default=-1, help='Index of the GPU to use')
parser.add_argument('--vram_thresh', type=int, default=1000, help='how many MB used is tolerated for the GPU before using the GPU')
parser.add_argument('--name', type=str, default="", help="Name for wandb run")
parser.add_argument('--tags', type=str, nargs='*', default=[], help="Tags for wandb run")
parser.add_argument('--notes', type=str, default="", help="Notes for wandb run")
parser.add_argument('--forceseed', type=int, default=-1, help='Overwrite seed if not set to default -1 for random seed')
parser.add_argument('--datasetseed', type=int, default=-1, help='Overwrite seed if not set to default -1 for random seed')

# dataset
parser.add_argument('--dataset', type=str, default="chinese_wikihan2022", help="Dataset to use", choices=[
    'chinese_baxter', 
    'chinese_wikihan2022', 
    'chinese_wikihan2022_augmented', 
    'Nromance_ipa', 
    'Nromance_orto', 
    'chinese_wikihan2022_augmented_drop_p_rate0.9'
])
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--test_val_batch_size', type=int, default=64, help="Batch size for val and test")
parser.add_argument('--min_daughters', type=int, default=1, help="Minimum number of daughters")
parser.add_argument('--skip_daughter_tone', type=str2bool, default=False, help="Whether to skip daughter tone")
parser.add_argument('--skip_protoform_tone', type=str2bool, default=False, help="Whether to skip protoform tone")
parser.add_argument('--d2p_use_lang_separaters', type=str2bool, default=True, help="Whether to use language separators for d2p") # this doesn't matter for p2d
parser.add_argument('--p2d_all_lang_summary_only', type=str2bool, default=True, help="Whether to use all language summary for p2d") # this doesn't matter for d2p
parser.add_argument('--proportion_labelled', type=float, default=0.1, help="Proportion of data to be labelled") # for p2d set this to 1.0

# training
parser.add_argument('--use_xavier_init', type=str2bool, default=True, help="Whether to use Xavier initialization")
parser.add_argument('--check_val_every_n_epoch', type=int, default=5, help="Frequency of validation checks")
parser.add_argument('--early_stopping_patience', type=int, default=99999, help="num val epoch without improvement to stop training")
parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of epochs for training")
parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs")
parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam optimizer")
parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam optimizer")
parser.add_argument('--eps', type=float, default=1e-8, help="Epsilon for Adam optimizer")
parser.add_argument('--lr', type=float, default=0.0003, help="Learning rate for training")
parser.add_argument('--weight_decay', type=float, default=0.0) # only for transformer

# strategy meta
parser.add_argument('--architecture', type=str, choices=['GRU', 'Transformer'], default='GRU')
parser.add_argument('--strat', type=str, choices=['supervised_only', 'pimodel', 'bpall_cringe', 'pimodel_bpall_cringe'], default='supervised_only', help='Semisupervised strategy')
parser.add_argument('--bootstrapping', action='store_true', default=False, help='Use bootstrapping')

# strategy specific
# pi model
parser.add_argument('--pi_consistency_type', type=str, choices=['mse', 'kl'], default='mse', help='Consistency type for pi-model. Choose between "mse" (mean squared error) and "kl" (Kullback-Leibler divergence).')
parser.add_argument('--pi_consistency_rampup_length', type=int, default=5)
parser.add_argument('--pi_max_consistency_scaling', type=float, default=100.0)

# bpall
parser.add_argument('--d2p_recon_loss_weight', type=float, default = 1.0,)
parser.add_argument('--d2p_kl_loss_weight', type=float, default = 0.5,)
parser.add_argument('--emb_pred_loss_weight', type=float, default = 0.01,)
parser.add_argument('--p2d_loss_on_gold_weight', type=float, default = 0.01,)
parser.add_argument('--p2d_loss_on_pred_weight', type=float, default = 1.0,)
parser.add_argument('--cringe_alpha', type=float, default = 0.0,)
parser.add_argument('--cringe_k', type=int, default = 5,)

# bootstrapping
parser.add_argument('--bootstrapping_min_epoch', type=int, default = 1)
parser.add_argument('--bootstrapping_log_prob_thresh', type=float, default = -0.006) # usually -0.001?
parser.add_argument('--bootstrapping_pseudolabelling_cap', type=int, default = 100)
parser.add_argument('--bootstrapping_inference_batch_size', type=int, default = 128)
parser.add_argument('--bootstrapping_alpha', type=float, default = 1.0)

# model specific
# Arguments for both d2p GRU and Transformer
parser.add_argument('--d2p_num_encoder_layers', type=int, default=2)
parser.add_argument('--d2p_dropout_p', type=float, default=0.30)
parser.add_argument('--d2p_inference_decode_max_length', type=int, default=12)
parser.add_argument('--d2p_feedforward_dim', type=int, default=512)
parser.add_argument('--d2p_embedding_dim', type=int, default=320) # gets overwritten if universal embedding on

# Arguments for d2p Transformer
parser.add_argument('--d2p_num_decoder_layers', type=int, default=2)
parser.add_argument('--d2p_nhead', type=int, default=8)
parser.add_argument('--d2p_max_len', type=int, default=128)

# Arguments for d2p GRU
parser.add_argument('--d2p_use_vae_latent', type=str2bool, default=False)
parser.add_argument('--d2p_use_bidirectional_encoder', type=str2bool, default=False)
parser.add_argument('--d2p_decode_mode', type=str, default='greedy_search', choices=['greedy_search', 'beam_search'])
parser.add_argument('--d2p_beam_search_alpha', type=float, default=1.0)
parser.add_argument('--d2p_beam_size', type=int, default=3)
parser.add_argument('--d2p_lang_embedding_when_decoder', type=str2bool, default=True)
parser.add_argument('--d2p_model_size', type=int, default=128)

# Arguments for both p2d GRU and Transformer
parser.add_argument('--p2d_num_encoder_layers', type=int, default=2)
parser.add_argument('--p2d_dropout_p', type=float, default=0.30)
parser.add_argument('--p2d_inference_decode_max_length', type=int, default=12)
parser.add_argument('--p2d_feedforward_dim', type=int, default=512)
parser.add_argument('--p2d_embedding_dim', type=int, default=320) # gets overwritten if universal embedding on

# Arguments for p2d GRU 
parser.add_argument('--p2d_use_vae_latent', type=str2bool, default=False)
parser.add_argument('--p2d_use_bidirectional_encoder', type=str2bool, default=True)
parser.add_argument('--p2d_decode_mode', type=str, default='greedy_search', choices=['greedy_search', 'beam_search'])
parser.add_argument('--p2d_beam_search_alpha', type=float, default=1.0)
parser.add_argument('--p2d_beam_size', type=int, default=5)
parser.add_argument('--p2d_lang_embedding_when_decoder', type=str2bool, default=False)
parser.add_argument('--p2d_prompt_mlp_with_one_hot_lang', type=str2bool, default=False)
parser.add_argument('--p2d_gated_mlp_by_target_lang', type=str2bool, default=False)
parser.add_argument('--p2d_model_size', type=int, default=128)

# Arguments for p2d Transformer
parser.add_argument('--p2d_num_decoder_layers', type=int, default=2)
parser.add_argument('--p2d_nhead', type=int, default=8)
parser.add_argument('--p2d_max_len', type=int, default=128)

# Arguments for bidirectional bridge
parser.add_argument('--universal_embedding', type=str2bool, default=True) # will be forced to be true for bpall strat. setting it false will not do anything for bpall.
parser.add_argument('--universal_embedding_dim', type=int, default=320)

args = parser.parse_args()
print("== parsed arguments ==")
pprint(vars(args))

seed = random.randint(0, 4294967295) # range accepted by numpy
datasetseed = random.randint(0, 4294967295)
if args.forceseed != -1:
    seed = args.forceseed
if args.datasetseed != -1:
    datasetseed = args.datasetseed

set_devices = 'auto' if args.cpu else (args.gpu if args.gpu != -1 else lib.getfreegpu.assign_free_gpus(threshold_vram_usage=args.vram_thresh, max_gpus=1, wait=True, sleep_time=10))
set_accelerator = 'cpu' if args.cpu else 'cuda'

print("== device setup ==")
pprint({
    "set_devices": set_devices,
    "set_accelerator": set_accelerator,
})

strat_id = get_strat_id(args.strat, args.bootstrapping)
strat_archi_id = get_strat_archi_id(args.strat, args.bootstrapping, args.architecture)
wandb_tags = args.tags
# endregion


# region === hyperparameters ===
strategy_configs = {
    'bpall_cringe': {
        'strategy_class_name': 'GreedySampleCringeBackpropThroughout',
        'has_p2d': True,
        'strategy_kwargs': {
            'd2p_recon_loss_weight': args.d2p_recon_loss_weight,
            'd2p_kl_loss_weight': args.d2p_kl_loss_weight,
            'emb_pred_loss_weight': args.emb_pred_loss_weight,
            'p2d_loss_on_gold_weight': args.p2d_loss_on_gold_weight,
            'p2d_loss_on_pred_weight': args.p2d_loss_on_pred_weight,
            'cringe_alpha': args.cringe_alpha,
            'cringe_k': args.cringe_k,
            'alignment_convolution_masking': False, # not planned, keep it simple
            'convolution_masking_residue': 0.2, # not planned, keep it simple
            'enable_pi_model': False,
            'pi_consistency_type': None,
            'pi_consistency_rampup_length': None,
            'pi_max_consistency_scaling': None,
            'pi_proportion_labelled': None,
        },
        'strategy_checkpoint_method': None if args.sweeping else ModelCheckpoint(
            monitor=f"d2p/val/phoneme_edit_distance",
            mode="min",
            save_top_k=1,
            verbose=args.dev,
        ),
        'early_stopping_method': EarlyStopping(
            monitor=f"d2p/val/phoneme_edit_distance", 
            mode="min",
            patience = args.early_stopping_patience,
        ),
    },
    'pimodel_bpall_cringe': {
        'strategy_class_name': 'GreedySampleCringeBackpropThroughout',
        'has_p2d': True,
        'strategy_kwargs': {
            'd2p_recon_loss_weight': args.d2p_recon_loss_weight,
            'd2p_kl_loss_weight': args.d2p_kl_loss_weight,
            'emb_pred_loss_weight': args.emb_pred_loss_weight,
            'p2d_loss_on_gold_weight': args.p2d_loss_on_gold_weight,
            'p2d_loss_on_pred_weight': args.p2d_loss_on_pred_weight,
            'cringe_alpha': args.cringe_alpha,
            'cringe_k': args.cringe_k,
            'alignment_convolution_masking': False, # not planned, keep it simple
            'convolution_masking_residue': 0.2, # not planned, keep it simple
            'enable_pi_model': True,
            'pi_consistency_type': args.pi_consistency_type,
            'pi_consistency_rampup_length': args.pi_consistency_rampup_length,
            'pi_max_consistency_scaling': args.pi_max_consistency_scaling,
            'pi_proportion_labelled': args.proportion_labelled,
        },
        'strategy_checkpoint_method': None if args.sweeping else ModelCheckpoint(
            monitor=f"d2p/val/phoneme_edit_distance",
            mode="min",
            save_top_k=1,
            verbose=args.dev,
        ),
        'early_stopping_method': EarlyStopping(
            monitor=f"d2p/val/phoneme_edit_distance", 
            mode="min",
            patience = args.early_stopping_patience,
        ),
    },
    'pimodel': {
        'strategy_class_name': 'PiModelD2P',
        'has_p2d': False,
        'strategy_kwargs': {
            'consistency_type': args.pi_consistency_type,
            'consistency_rampup_length': args.pi_consistency_rampup_length,
            'max_consistency_scaling': args.pi_max_consistency_scaling,
            'proportion_labelled': args.proportion_labelled,
        },
        'strategy_checkpoint_method': None if args.sweeping else ModelCheckpoint(
            monitor=f"d2p/val/phoneme_edit_distance",
            mode="min",
            save_top_k=1,
            verbose=args.dev,
        ),
        'early_stopping_method': EarlyStopping(
            monitor=f"d2p/val/phoneme_edit_distance", 
            mode="min",
            patience = args.early_stopping_patience,
        ),
    },
    'supervised_only': {
        'strategy_class_name': 'SupervisedOnlyD2P',
        'has_p2d': False,
        'strategy_kwargs': {
        },
        'strategy_checkpoint_method': None if args.sweeping else ModelCheckpoint(
            monitor=f"d2p/val/phoneme_edit_distance",
            mode="min",
            save_top_k=1,
            verbose=args.dev,
        ),
        'early_stopping_method': EarlyStopping(
            monitor=f"d2p/val/phoneme_edit_distance", 
            mode="min",
            patience = args.early_stopping_patience,
        ),
    },
}
bootstrapping_config = {
    'bootstrapping_min_epoch': args.bootstrapping_min_epoch,
    'bootstrapping_log_prob_thresh': args.bootstrapping_log_prob_thresh,
    'bootstrapping_pseudolabelling_cap': args.bootstrapping_pseudolabelling_cap,
    'bootstrapping_inference_batch_size': args.bootstrapping_inference_batch_size,
    'bootstrapping_alpha': args.bootstrapping_alpha,
}
config = {
    'seed': seed,
    'datasetseed': datasetseed,
    
    'architecture': args.architecture,
    
    'dataset': args.dataset,
    'batch_size': args.batch_size,
    'test_val_batch_size': args.test_val_batch_size,
    'min_daughters': args.min_daughters, 
    'skip_daughter_tone': args.skip_daughter_tone,
    'skip_protoform_tone': args.skip_protoform_tone,

    'd2p_use_lang_separaters': args.d2p_use_lang_separaters, 
    'p2d_all_lang_summary_only': args.p2d_all_lang_summary_only, 
    'proportion_labelled': args.proportion_labelled, 
    'transformer_d2p_d_cat_style': args.architecture == 'Transformer',

    'use_xavier_init': args.use_xavier_init,
    'lr': args.lr,
    'max_epochs': args.max_epochs,
    'warmup_epochs': args.warmup_epochs,
    'check_val_every_n_epoch': args.check_val_every_n_epoch,
    'early_stopping_patience': args.early_stopping_patience,
    'beta1': args.beta1,
    'beta2': args.beta2,
    'eps': args.eps,
    'weight_decay': args.weight_decay,
    
    'strat': args.strat,
    'strat_id': strat_id.value,
    'strat_archi_id': strat_archi_id,
}

GRU_d2p_config = {    
    'd2p_num_encoder_layers': args.d2p_num_encoder_layers,
    'd2p_dropout_p': args.d2p_dropout_p,
    'd2p_use_vae_latent': args.d2p_use_vae_latent,
    'd2p_inference_decode_max_length': args.d2p_inference_decode_max_length,
    'd2p_use_bidirectional_encoder': args.d2p_use_bidirectional_encoder,
    'd2p_decode_mode': args.d2p_decode_mode,
    'd2p_beam_search_alpha': args.d2p_beam_search_alpha,
    'd2p_beam_size': args.d2p_beam_size,
    'd2p_lang_embedding_when_decoder': args.d2p_lang_embedding_when_decoder,
    'd2p_feedforward_dim': args.d2p_feedforward_dim,
    'd2p_embedding_dim': args.d2p_embedding_dim,
    'd2p_model_size': args.d2p_model_size,
}
GRU_p2d_config = {
    'p2d_num_encoder_layers': args.p2d_num_encoder_layers,
    'p2d_dropout_p': args.p2d_dropout_p,
    'p2d_use_vae_latent': args.p2d_use_vae_latent,
    'p2d_inference_decode_max_length': args.p2d_inference_decode_max_length,
    'p2d_use_bidirectional_encoder': args.p2d_use_bidirectional_encoder,
    'p2d_decode_mode': args.p2d_decode_mode,
    'p2d_beam_search_alpha': args.p2d_beam_search_alpha,
    'p2d_beam_size': args.p2d_beam_size,
    'p2d_lang_embedding_when_decoder': args.p2d_lang_embedding_when_decoder,
    'p2d_prompt_mlp_with_one_hot_lang': args.p2d_prompt_mlp_with_one_hot_lang,
    'p2d_gated_mlp_by_target_lang': args.p2d_gated_mlp_by_target_lang,
    'p2d_feedforward_dim': args.p2d_feedforward_dim,
    'p2d_embedding_dim': args.p2d_embedding_dim,
    'p2d_model_size': args.p2d_model_size,
}
Transformer_d2p_config = { # same as Kim et al for romance
    'd2p_num_encoder_layers': args.d2p_num_encoder_layers,
    'd2p_num_decoder_layers': args.d2p_num_decoder_layers,
    'd2p_embedding_dim': args.d2p_embedding_dim,
    'd2p_nhead': args.d2p_nhead,
    'd2p_feedforward_dim': args.d2p_feedforward_dim,
    'd2p_dropout_p': args.d2p_dropout_p,
    'd2p_max_len': args.d2p_max_len,
    'd2p_inference_decode_max_length': args.d2p_inference_decode_max_length,
}
Transformer_p2d_config = {
    'p2d_num_encoder_layers': args.p2d_num_encoder_layers,
    'p2d_num_decoder_layers': args.p2d_num_decoder_layers,
    'p2d_embedding_dim': args.p2d_embedding_dim,
    'p2d_nhead': args.p2d_nhead,
    'p2d_feedforward_dim': args.p2d_feedforward_dim,
    'p2d_dropout_p': args.p2d_dropout_p,
    'p2d_max_len': args.p2d_max_len,
    'p2d_inference_decode_max_length': args.p2d_inference_decode_max_length,
}
bidir_config = {
    'universal_embedding': args.universal_embedding,
    'universal_embedding_dim': args.universal_embedding_dim,
}
match args.architecture, args.bootstrapping:
    case ('GRU', False):
        config = {**config, **GRU_d2p_config, **GRU_p2d_config, **bidir_config}
    case ('GRU', True):
        config = {**config, **GRU_d2p_config, **GRU_p2d_config, **bidir_config, **bootstrapping_config}
    case ('Transformer', False):
        config = {**config, **Transformer_d2p_config, **Transformer_p2d_config, **bidir_config}
    case ('Transformer', True):
        config = {**config, **Transformer_d2p_config, **Transformer_p2d_config, **bidir_config, **bootstrapping_config}
config['strategy_config'] = strategy_configs[args.strat]
print("config:", config)
# endregion


# region === wandb ===
if args.dev:
    wandb_tags.append('dev')
if args.bootstrapping:
    wandb_tags.append('bootstrapping')
wandb_tags.append(args.architecture)
wandb_tags.append(config['strat'])
wandb_tags.append(str(config['proportion_labelled']) + 'labelled')

run = wandb.init(
    mode = "disabled" if (args.nowandb or args.countparam) else "online",
    entity = WANDB_ENTITY,
    project = WANDB_PROJECT, 
    config = config,
    tags=wandb_tags,
    notes=args.notes,
    allow_val_change=True,
    name=args.name,
)
wandb_logger = WandbLogger(
    log_model = args.logmodel,
    experiment = run,
)
if (not args.sweeping) and (not args.logmodel):
    print("WARNING: not sweeping and not saving models. make sure to not do this for actual experiments")
print("== wandb.config ==")
pprint(dict(wandb.config))

wandb.define_metric(f'd2p/val/accuracy', summary='max')
wandb.define_metric(f'd2p/val/bcubed_f_score', summary='max')
wandb.define_metric(f'd2p/val/phoneme_error_rate', summary='min')
wandb.define_metric(f'd2p/val/phoneme_edit_distance', summary='min')
wandb.define_metric(f'd2p/val/feature_error_rate', summary='min')

wandb.define_metric(f'd2p/test/accuracy')
wandb.define_metric(f'd2p/test/bcubed_f_score')
wandb.define_metric(f'd2p/test/phoneme_error_rate')
wandb.define_metric(f'd2p/test/phoneme_edit_distance')
wandb.define_metric(f'd2p/test/feature_error_rate')
# endregion


# region === loading dataset and model ===
dm = DataloaderManager(
    data_dir = f"data/{wandb.config.dataset}", 
    
    batch_size = wandb.config.batch_size,
    test_val_batch_size = wandb.config.test_val_batch_size,
    shuffle_train = True,
    
    lang_separators = wandb.config.d2p_use_lang_separaters,
    skip_daughter_tone = wandb.config.skip_daughter_tone,
    skip_protoform_tone = wandb.config.skip_protoform_tone,
    include_lang_tkns_in_ipa_vocab = True,
    transformer_d2p_d_cat_style = wandb.config.transformer_d2p_d_cat_style,
    
    daughter_subset = None, 
    min_daughters = wandb.config.min_daughters,
    verbose = False,
    
    proportion_labelled = wandb.config.proportion_labelled, 
    datasetseed = datasetseed,
)
wandb.log({'train_p_labelled_mask_fingerprint': dm.train_p_labelled_mask_fingerprint})
pl.seed_everything(seed)
match wandb.config.architecture:
    case 'GRU':
        bidir_model = biDirReconModelRNN(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab, 
            has_p2d = wandb.config.strategy_config['has_p2d'],
            
            d2p_num_encoder_layers = wandb.config.d2p_num_encoder_layers,
            d2p_dropout_p = wandb.config.d2p_dropout_p,
            d2p_use_vae_latent = wandb.config.d2p_use_vae_latent,
            d2p_inference_decode_max_length = wandb.config.d2p_inference_decode_max_length,
            d2p_use_bidirectional_encoder = wandb.config.d2p_use_bidirectional_encoder,
            d2p_decode_mode = wandb.config.d2p_decode_mode,
            d2p_beam_search_alpha = wandb.config.d2p_beam_search_alpha,
            d2p_beam_size = wandb.config.d2p_beam_size,
            d2p_lang_embedding_when_decoder = wandb.config.d2p_lang_embedding_when_decoder,
            
            p2d_num_encoder_layers = wandb.config.p2d_num_encoder_layers,
            p2d_dropout_p = wandb.config.p2d_dropout_p,
            p2d_use_vae_latent = wandb.config.p2d_use_vae_latent,
            p2d_inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            p2d_use_bidirectional_encoder = wandb.config.p2d_use_bidirectional_encoder,
            p2d_decode_mode = wandb.config.p2d_decode_mode,
            p2d_beam_search_alpha = wandb.config.p2d_beam_search_alpha,
            p2d_beam_size = wandb.config.p2d_beam_size,
            p2d_lang_embedding_when_decoder = wandb.config.p2d_lang_embedding_when_decoder,
            p2d_prompt_mlp_with_one_hot_lang = wandb.config.p2d_prompt_mlp_with_one_hot_lang,
            p2d_gated_mlp_by_target_lang = wandb.config.p2d_gated_mlp_by_target_lang,
            p2d_all_lang_summary_only = True,

            d2p_feedforward_dim = wandb.config.d2p_feedforward_dim, 
            d2p_embedding_dim = wandb.config.d2p_embedding_dim,
            d2p_model_size = wandb.config.d2p_model_size,
            
            p2d_feedforward_dim = wandb.config.p2d_feedforward_dim, 
            p2d_embedding_dim = wandb.config.p2d_embedding_dim,
            p2d_model_size = wandb.config.p2d_model_size,

            use_xavier_init = True,
            lr = wandb.config.lr,
            max_epochs = wandb.config.max_epochs,
            warmup_epochs = wandb.config.warmup_epochs,
            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
            weight_decay = wandb.config.weight_decay,
            
            universal_embedding=wandb.config.universal_embedding,
            universal_embedding_dim=wandb.config.universal_embedding_dim,
            
            strategy=getattr(models.biDirReconStrategies, wandb.config.strategy_config['strategy_class_name'])(**wandb.config.strategy_config['strategy_kwargs']),
        )
    case 'Transformer':
        bidir_model = biDirReconModelTrans(
            ipa_vocab = dm.ipa_vocab,
            lang_vocab = dm.lang_vocab, 
            has_p2d = wandb.config.strategy_config['has_p2d'],

            d2p_num_encoder_layers = wandb.config.d2p_num_encoder_layers,
            d2p_num_decoder_layers = wandb.config.d2p_num_decoder_layers,
            d2p_nhead = wandb.config.d2p_nhead,
            d2p_dropout_p = wandb.config.d2p_dropout_p,
            d2p_inference_decode_max_length = wandb.config.d2p_inference_decode_max_length,
            d2p_max_len = wandb.config.d2p_max_len,
            d2p_feedforward_dim = wandb.config.d2p_feedforward_dim,
            d2p_embedding_dim = wandb.config.d2p_embedding_dim,

            p2d_num_encoder_layers = wandb.config.p2d_num_encoder_layers,
            p2d_num_decoder_layers = wandb.config.p2d_num_decoder_layers,
            p2d_nhead = wandb.config.p2d_nhead,
            p2d_dropout_p = wandb.config.p2d_dropout_p,
            p2d_inference_decode_max_length = wandb.config.p2d_inference_decode_max_length,
            p2d_max_len = wandb.config.p2d_max_len,
            p2d_feedforward_dim = wandb.config.p2d_feedforward_dim,
            p2d_embedding_dim = wandb.config.p2d_embedding_dim,
            p2d_all_lang_summary_only = wandb.config.p2d_all_lang_summary_only,

            use_xavier_init = True,
            lr = wandb.config.lr,
            max_epochs = wandb.config.max_epochs,
            warmup_epochs = wandb.config.warmup_epochs,
            beta1 = wandb.config.beta1,
            beta2 = wandb.config.beta2,
            eps = wandb.config.eps,
            weight_decay = wandb.config.weight_decay,
            
            universal_embedding=wandb.config.universal_embedding,
            universal_embedding_dim=wandb.config.universal_embedding_dim,
            
            strategy=getattr(models.biDirReconStrategies, wandb.config.strategy_config['strategy_class_name'])(**wandb.config.strategy_config['strategy_kwargs']),
        )
    case _:
        assert False, "ahh?"
# endregion



# region === training ===
callbacks=[]
if 'strategy_checkpoint_method' in strategy_configs[wandb.config['strat']] and strategy_configs[wandb.config['strat']]['strategy_checkpoint_method'] is not None:
    callbacks.append(strategy_configs[wandb.config['strat']]['strategy_checkpoint_method'])
if 'early_stopping_method' in strategy_configs[wandb.config['strat']] and strategy_configs[wandb.config['strat']]['early_stopping_method'] is not None:
    callbacks.append(strategy_configs[wandb.config['strat']]['early_stopping_method'])

if args.bootstrapping:
    callbacks.append(Pseudolabeller(
        min_epoch = wandb.config.bootstrapping_min_epoch,
        log_prob_thresh = wandb.config.bootstrapping_log_prob_thresh,
        pseudolabelling_cap = wandb.config.bootstrapping_pseudolabelling_cap,
        inference_batch_size = wandb.config.bootstrapping_inference_batch_size,
        alpha = wandb.config.bootstrapping_alpha,
    ))


trainer = pl.Trainer(
    accelerator=set_accelerator,
    devices=set_devices,
    max_epochs=wandb.config.max_epochs,
    logger=wandb_logger,
    log_every_n_steps=10,
    check_val_every_n_epoch=1 if args.dev else wandb.config.check_val_every_n_epoch,
    callbacks = callbacks,
    num_sanity_val_steps = 0 if args.dev else 2,
    enable_progress_bar = not args.noprogressbar,
)
# so that we can find it when making subtrainers
trainer.set_devices = set_devices
trainer.set_accelerator = set_accelerator

dm.batch_size = wandb.config.batch_size

if args.wandbwatch:
    wandb_logger.watch(bidir_model)

if args.countparam:
    print(f"{args.dataset} {wandb.config.strat_archi_id} param count isss {utils.count_parameters(bidir_model)}")
else:
    trainer.fit(
        bidir_model, 
        train_dataloaders = dm.train_dataloader(), 
        val_dataloaders=dm.val_dataloader()
    )

    if not args.sweeping:
        trainer.test(bidir_model, ckpt_path="best", dataloaders=dm.test_dataloader())
        
        bidir_model.transductive_test = True
        trainer.test(bidir_model, ckpt_path="best", dataloaders=dm.train_transductive_test_dataloader())
# endregion
