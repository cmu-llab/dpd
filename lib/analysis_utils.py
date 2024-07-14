# region imports
from __future__ import annotations
import pickle
from typing import TYPE_CHECKING
from lib import getfreegpu
from lib.dataset import DatasetConcat
from models.biDirReconIntegration import biDirReconModelRNN, biDirReconModelTrans
import models.biDirReconStrategies
from sklearn.cluster import AgglomerativeClustering, ward_tree
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.cluster import hierarchy
import models.utils as utils
import scipy
import matplotlib
from models.encoderDecoderTransformer import EncoderDecoderTransformer
import lingpy
if TYPE_CHECKING:
    from models.biDirReconStrategies import BidirTrainStrategyBase
from matplotlib import pyplot as plt
import wandb
api = wandb.Api()
from lib.dataloader_manager import DataloaderManager
from models.encoderDecoderRNN import Seq2SeqRNN
import torch
import pytorch_lightning as pl
from torch import Tensor
from specialtokens import *
from torch.nn import functional as F
from tqdm.notebook import tqdm
from einops import rearrange, repeat
import pandas as pd
from lib.tensor_utils import num_sequences_equal, sequences_equal
import seaborn as sns
from prelude import *
import warnings
warnings.filterwarnings(action="ignore", message=".*num_workers.*")
warnings.filterwarnings(action="ignore", message=".*negatively affect performance.*")
warnings.filterwarnings(action="ignore", message=".*MPS available.*")
warnings.filterwarnings(action="ignore", message=".*or one of them to enable TensorBoard support by default.*")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from collections import Counter

# endregion


TEST_VAL_BATCH_SIZE = 64 # unified batch size for analysis inference

def is_sorted_dec(l):
    return all(l[i] >= l[i+1] for i in range(len(l) - 1))

# trick to get config from dictionary with . syntax
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# note this runs validation loop, so things are prefixed by val. ignore that.
def eval_on_set(submodel, dm, split: str):
    evaluator = pl.Trainer(
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices= getfreegpu.assign_free_gpus(threshold_vram_usage=2000, max_gpus=1, wait=True, sleep_time=10) if torch.cuda.is_available() else 'auto',
        max_epochs=1,
    )
    match split:
        case 'val':
            res = evaluator.test(submodel, dataloaders=dm.val_dataloader(), verbose=False)
        case 'train':
            res = evaluator.test(submodel, dataloaders=dm.train_dataloader(), verbose=False)
        case 'test':
            res = evaluator.test(submodel, dataloaders=dm.test_dataloader(), verbose=False)
        case _:
            raise ValueError(f"Unknown split: {split}")
    return res[0]

def get_sample_batch(dm, split: str):
    if split == 'test':
        for batch in dm.test_dataloader():       
            return batch
    if split == 'train':
        for batch in dm.train_dataloader():       
            return batch
    if split == 'val':
        for batch in dm.val_dataloader():       
            return batch
    raise ValueError(f"Unknown split: {split}")

def get_accuracy(model, dm: DataloaderManager, split: str):
    evaluator = pl.Trainer(accelerator='cpu', max_epochs=1, enable_progress_bar=False)
    loader = get_loader_for_split(dm, split)
    accuracy = evaluator.validate(model, dataloaders=loader, verbose=False)[0][f'{model.logger_prefix}/val/accuracy']
    return accuracy

def get_loader_for_split(dm, split):
    match split:
        case 'train':
            loader = dm.train_dataloader()
        case 'val':
            loader = dm.val_dataloader()
        case 'test':
            loader = dm.test_dataloader()
        case _:
            raise ValueError(f'Invalid split: {split}')
    return loader

def sample_output_for_run(dataset: DatasetConcat, run):
    samples = []
    for i in range(dataset.length):
        minimal_singleton_batch = dataset.collate_fn([dataset[i]])
        
        match run.config_class.architecture:
            case 'GRU':
                source_tokens, source_langs, source_seqs_lens, target_tokens, _target_lang_ipa_ids, target_lang_lang_ids = run.model.d2p.unpack_batch(minimal_singleton_batch)
                
                N = 1
                target_lang_lang_ids = repeat((torch.LongTensor([run.model.d2p.lang_vocab.get_idx(run.model.d2p.protolang)]).to(run.model.d2p.device)), '1 -> N 1', N=N)

                prediction = run.model.d2p.greedy_decode(
                    source_tokens=source_tokens,
                    source_langs=source_langs,
                    source_seqs_lens=source_seqs_lens,
                    target_langs=target_lang_lang_ids,
                )

            case 'Transformer':
                N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(minimal_singleton_batch, run.model.d2p.device, run.model.d2p.task, run.model.d2p.ipa_vocab, run.model.d2p.lang_vocab, run.model.d2p.protolang)
                
                prediction = run.model.d2p.greedy_decode(s_tkns, s_indv_lens, s_langs, s_mask, s_pad_mask, decode_max_len=run.model.d2p.inference_decode_max_length) 
                
            case _:
                raise Match

        samples.append(
            {
                **dataset.D[i],
                **dataset.Pl[i],
                "proto_hat": run.model.d2p.ipa_vocab.to_tokens(prediction[0]),
            }
        )
        
    samples_df = pd.DataFrame(samples)
    return samples_df


class BidirRun:
    def __init__(self,
        run_id: str, 
        version: str = 'best', 
        verbose: bool = False,
        entity: str = 'wandbentity',
        project: str = 'wandbproject',
        
        # only if loading checkpoint from file
        from_file: bool = False,
        checkpoint_filepath: str = None,
        config_filepath: str = None,
    ):
        self.from_file = from_file
        self.checkpoint_filepath = checkpoint_filepath
        self.config_filepath = config_filepath
        if not from_file:
            artifact = api.artifact(f'{entity}/{project}/model-{run_id}:{version}')
                
            artifact_dir = artifact.download()
            if verbose: print(f"artifact sits in: {artifact_dir}")
            run = api.run(f'{entity}/{project}/{run_id}')
            config_dict = run.config
        else:
            assert checkpoint_filepath is not None
            assert config_filepath is not None
            artifact_dir = checkpoint_filepath
            artifact = None
            run = None
            with open(config_filepath, 'rb') as f:
                config_dict = pickle.load(f)
        
        
        config_dict["strategy_config"]["early_stopping_method"] = None
        config_dict["strategy_config"]["strategy_checkpoint_method"] = None
            
        config_class = AttributeDict(config_dict)
                
        self.artifact, self.artifact_dir, self.run, self.config_dict, self.config_class = artifact, artifact_dir, run, config_dict, config_class
        
        self.dm = self.get_dm()
        self.model = self.get_model(self.dm)
        self.ipa_vocab = self.dm.ipa_vocab
        self.lang_vocab = self.dm.lang_vocab
        
        match self.config_class.architecture:
            case 'Transformer':
                self.ipa_emb_tensor = self.model.d2p.ipa_embedding.embedding.weight.data
                self.lang_emb_tensor = self.model.d2p.lang_embedding.embedding.weight.data
            case 'GRU':
                self.ipa_emb_tensor = self.model.d2p.embeddings.char_embeddings.weight.data
                self.lang_emb_tensor = self.model.d2p.embeddings.lang_embeddings.weight.data
            case _:
                assert False
        
    def get_dm(self):
        if not self.from_file:
            print(f"INFO: they fingerprint should end up being {self.run.summary['train_p_labelled_mask_fingerprint']}")
        c = self.config_class
        dm = DataloaderManager(
            data_dir = f"data/{c.dataset}", 
            
            batch_size = c.batch_size,
            test_val_batch_size = c.test_val_batch_size,
            shuffle_train = True,
            
            lang_separators = c.d2p_use_lang_separaters,
            skip_daughter_tone = c.skip_daughter_tone,
            skip_protoform_tone = c.skip_protoform_tone,
            include_lang_tkns_in_ipa_vocab = True,
            transformer_d2p_d_cat_style = c.transformer_d2p_d_cat_style,
            
            daughter_subset = None, 
            min_daughters = c.min_daughters,
            verbose = False,
            
            proportion_labelled = c.proportion_labelled, 
            datasetseed = c.datasetseed,
        )
        return dm

    def get_model(self, dm: DataloaderManager):
        c = self.config_class
        
        match c.architecture:
            case 'GRU':
                model = biDirReconModelRNN.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),

                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab, 
                    has_p2d = c.strategy_config['has_p2d'],
                    
                    d2p_num_encoder_layers = c.d2p_num_encoder_layers,
                    d2p_dropout_p = c.d2p_dropout_p,
                    d2p_use_vae_latent = c.d2p_use_vae_latent,
                    d2p_inference_decode_max_length = c.d2p_inference_decode_max_length,
                    d2p_use_bidirectional_encoder = c.d2p_use_bidirectional_encoder,
                    d2p_decode_mode = c.d2p_decode_mode,
                    d2p_beam_search_alpha = c.d2p_beam_search_alpha,
                    d2p_beam_size = c.d2p_beam_size,
                    d2p_lang_embedding_when_decoder = c.d2p_lang_embedding_when_decoder,
                    
                    p2d_num_encoder_layers = c.p2d_num_encoder_layers,
                    p2d_dropout_p = c.p2d_dropout_p,
                    p2d_use_vae_latent = c.p2d_use_vae_latent,
                    p2d_inference_decode_max_length = c.p2d_inference_decode_max_length,
                    p2d_use_bidirectional_encoder = c.p2d_use_bidirectional_encoder,
                    p2d_decode_mode = c.p2d_decode_mode,
                    p2d_beam_search_alpha = c.p2d_beam_search_alpha,
                    p2d_beam_size = c.p2d_beam_size,
                    p2d_lang_embedding_when_decoder = c.p2d_lang_embedding_when_decoder,
                    p2d_prompt_mlp_with_one_hot_lang = c.p2d_prompt_mlp_with_one_hot_lang,
                    p2d_gated_mlp_by_target_lang = c.p2d_gated_mlp_by_target_lang,
                    p2d_all_lang_summary_only = True,

                    d2p_feedforward_dim = c.d2p_feedforward_dim, 
                    d2p_embedding_dim = c.d2p_embedding_dim,
                    d2p_model_size = c.d2p_model_size,
                    
                    p2d_feedforward_dim = c.p2d_feedforward_dim, 
                    p2d_embedding_dim = c.p2d_embedding_dim,
                    p2d_model_size = c.p2d_model_size,

                    use_xavier_init = True,
                    lr = c.lr,
                    max_epochs = c.max_epochs,
                    warmup_epochs = c.warmup_epochs,
                    beta1 = c.beta1,
                    beta2 = c.beta2,
                    eps = c.eps,
                    weight_decay = c.weight_decay,
                    
                    universal_embedding=c.universal_embedding,
                    universal_embedding_dim=c.universal_embedding_dim,
                    
                    strategy=getattr(models.biDirReconStrategies, c.strategy_config['strategy_class_name'])(**c.strategy_config['strategy_kwargs']),
                )
            case 'Transformer':
                model = biDirReconModelTrans.load_from_checkpoint(
                    checkpoint_path=f"{self.artifact_dir}/model.ckpt",
                    map_location=torch.device('cpu'),

                    ipa_vocab = dm.ipa_vocab,
                    lang_vocab = dm.lang_vocab, 
                    has_p2d = c.strategy_config['has_p2d'],

                    d2p_num_encoder_layers = c.d2p_num_encoder_layers,
                    d2p_num_decoder_layers = c.d2p_num_decoder_layers,
                    d2p_nhead = c.d2p_nhead,
                    d2p_dropout_p = c.d2p_dropout_p,
                    d2p_inference_decode_max_length = c.d2p_inference_decode_max_length,
                    d2p_max_len = c.d2p_max_len,
                    d2p_feedforward_dim = c.d2p_feedforward_dim,
                    d2p_embedding_dim = c.d2p_embedding_dim,

                    p2d_num_encoder_layers = c.p2d_num_encoder_layers,
                    p2d_num_decoder_layers = c.p2d_num_decoder_layers,
                    p2d_nhead = c.p2d_nhead,
                    p2d_dropout_p = c.p2d_dropout_p,
                    p2d_inference_decode_max_length = c.p2d_inference_decode_max_length,
                    p2d_max_len = c.p2d_max_len,
                    p2d_feedforward_dim = c.p2d_feedforward_dim,
                    p2d_embedding_dim = c.p2d_embedding_dim,
                    p2d_all_lang_summary_only = c.p2d_all_lang_summary_only,

                    use_xavier_init = True,
                    lr = c.lr,
                    max_epochs = c.max_epochs,
                    warmup_epochs = c.warmup_epochs,
                    beta1 = c.beta1,
                    beta2 = c.beta2,
                    eps = c.eps,
                    weight_decay = c.weight_decay,
                    
                    universal_embedding=c.universal_embedding,
                    universal_embedding_dim=c.universal_embedding_dim,
                    
                    strategy=getattr(models.biDirReconStrategies, c.strategy_config['strategy_class_name'])(**c.strategy_config['strategy_kwargs']),
                )
            case _:
                assert False, "ahh?"
        
        model.eval()
        return model

    def evaluate_d2p(self, split: str):
        evaluator = pl.Trainer(accelerator='cpu', max_epochs=1, enable_progress_bar=False)
        loader = get_loader_for_split(self.dm, split)
        return evaluator.validate(self.model.d2p, dataloaders=loader, verbose=False)


    def get_accuracy(self, split: str):
        evaluator = pl.Trainer(accelerator='cpu', max_epochs=1, enable_progress_bar=False)
        loader = get_loader_for_split(self.dm, split)
        d2p_accuracy = evaluator.validate(self.model.d2p, dataloaders=loader, verbose=False)[0][f'{self.model.d2p.logger_prefix}/val/accuracy']
        if self.config_class.strategy_config['has_p2d']:
            p2d_accuracy = evaluator.validate(self.model.p2d, dataloaders=loader, verbose=False)[0][f'{self.model.p2d.logger_prefix}/val/accuracy']
        else: 
            p2d_accuracy = None
        
        return {
            "d2p_accuracy": d2p_accuracy,
            "p2d_accuracy": p2d_accuracy,
        }



# region d2p2d analysis consts
ALL_STRATS = ['BpallPiBst', 'BpallBst', 'BpallPi', 'Bpall', 'PiBst', 'SupvBst', 'Pi', 'Supv']
HAS_P2D_STRATS = ['BpallPiBst', 'BpallBst', 'BpallPi', 'Bpall']
HAS_P2D_STRAT_ARCHIS = ['GRUBpallPiBst', 'GRUBpallBst', 'GRUBpallPi', 'GRUBpall', 'TransBpallPiBst', 'TransBpallBst', 'TransBpallPi', 'TransBpall']
WEAK_BASELINE_STRATS = ['SupvBst', 'Pi', 'Supv']
STRONG_BASELINE_STRATS = ['SupvBst', 'Pi', 'Supv', 'PiBst']
ALL_STRAT_ARCHIS = ['TransBpallPi', 'TransSupvBst', 'TransBpallPiBst', 'TransPiBst', 'TransBpallBst', 'GRUBpallPiBst', 'GRUBpallBst', 'GRUPiBst', 'GRUSupvBst', 'TransBpall', 'TransPi', 'TransSupv', 'GRUBpallPi', 'GRUBpall', 'GRUPi', 'GRUSupv']
ALL_PORPORTIONS = [0.05, 0.1, 0.2, 0.3]
STRAT_ARCHIS_LATEX_NAMES = {
    'TransBpallPi': r'Trans-\BpallPiName{}', 
    'TransSupvBst': r'Trans-\SupvBstName{}', 
    'TransBpallPiBst': r'Trans-\BpallPiBstName{}', 
    'TransPiBst': r'Trans-\PiBstName{}', 
    'TransBpallBst': r'Trans-\BpallBstName{}', 
    'GRUBpallPiBst': r'GRU-\BpallPiBstName{}', 
    'GRUBpallBst': r'GRU-\BpallBstName{}', 
    'GRUPiBst': r'GRU-\PiBstName{}', 
    'GRUSupvBst': r'GRU-\SupvBstName{}', 
    'TransBpall': r'Trans-\BpallName{}', 
    'TransPi': r'Trans-\PiName{}', 
    'TransSupv': r'Trans-\SupvName{}', 
    'GRUBpallPi': r'GRU-\BpallPiName{}', 
    'GRUBpall': r'GRU-\BpallName{}', 
    'GRUPi': r'GRU-\PiName{}', 
    'GRUSupv': r'GRU-\SupvName{}', 
}
STRAT_ARCHIS_PLT_NAMES = {
    'TransBpallPiBst': r'Trans-DPD-ΠM-BST', 
    'TransBpallPi': r'Trans-DPD-ΠM', 
    'TransBpallBst': r'Trans-DPD-BST', 
    'TransBpall': r'Trans-DPD', 
    'TransPiBst': r'Trans-ΠM-BST', 
    'TransSupvBst': r'Trans-BST', 
    'TransPi': r'Trans-ΠM', 
    'TransSupv': r'Trans-SUPV', 
    'GRUBpallPiBst': r'GRU-DPD-ΠM-BST', 
    'GRUBpallPi': r'GRU-DPD-ΠM', 
    'GRUBpallBst': r'GRU-DPD-BST', 
    'GRUBpall': r'GRU-DPD', 
    'GRUPiBst': r'GRU-ΠM-BST', 
    'GRUSupvBst': r'GRU-BST', 
    'GRUPi': r'GRU-ΠM', 
    'GRUSupv': r'GRU-SUPV', 
}
DATASET_LATEX_NAMES = {
    'chinese_wikihan2022': r'WikiHan', 
    'Nromance_ipa': r'Rom-phon', 
}
GROUP_PLT_NAMES = {'group1': 'group 1','group2': 'group 2','group3': 'group 3','group4': 'group 4'}
PLOT_STRAT_ARCHI_ORD = [
    'TransBpallPiBst', 
    'TransBpallBst', 
    'TransBpallPi', 
    'TransBpall', 
    'TransPiBst', 
    'TransSupvBst', 
    'TransPi', 
    'TransSupv', 
    'GRUBpallPiBst', 
    'GRUBpallBst', 
    'GRUBpallPi', 
    'GRUBpall', 
    'GRUPiBst', 
    'GRUSupvBst', 
    'GRUPi', 
    'GRUSupv', 
]
STRAT_ARCHI_BPALLNONBPALL_ORD = [
    'TransBpallPiBst', 
    'TransPiBst', 
    'TransBpallBst', 
    'TransSupvBst', 
    'TransBpallPi', 
    'TransPi', 
    'TransBpall', 
    'TransSupv', 
    
    'GRUBpallPiBst', 
    'GRUPiBst', 
    'GRUBpallBst', 
    'GRUSupvBst', 
    'GRUBpallPi', 
    'GRUPi', 
    'GRUBpall', 
    'GRUSupv', 
]
# endregion
