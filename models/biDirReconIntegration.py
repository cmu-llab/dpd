from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, Optional
from models.partials.mlp import MLP

if TYPE_CHECKING:
    from models.biDirReconStrategies import BidirTrainStrategyBase

from models.encoderDecoderRNN import Seq2SeqRNN
from models.encoderDecoderTransformer import EncoderDecoderTransformer, TokenEmbedding

import torch
import pytorch_lightning as pl
from torch import Tensor
import transformers

from specialtokens import *

from models.partials.embedding import Embedding
from models.partials.embeddingPredictionNet import EmbeddingPredictionNet
from lib.vocab import Vocab
import wandb

class biDirReconModelBase(pl.LightningModule):
    def __init__(self,
                
        ipa_vocab: Vocab, 
        lang_vocab: Vocab,
        has_p2d: bool, 
                
        use_xavier_init: bool,
        lr: float,
        max_epochs: int,
        warmup_epochs: int,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,

        universal_embedding: bool, # whether to use the same embedding for both d2p and p2d
        universal_embedding_dim: int | None,
        
        strategy: BidirTrainStrategyBase | None, # can be none if not training
    ) -> None:
        super().__init__()
        
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_xavier_init = use_xavier_init
        self.weight_decay = weight_decay
        
        self.ipa_vocab = ipa_vocab
        self.lang_vocab = lang_vocab
        self.protolang = lang_vocab.protolang
    
        self.has_p2d = has_p2d
        self.embeddings: Embedding | None
        self.embeddings = None 
        self.universal_embedding = universal_embedding
        self.universal_embedding_dim = universal_embedding_dim

        self.evaluation_step_outputs = {}
        self.samples_tables: dict[str, Optional[wandb.Table]] = {
            'val': None,
            'test': None,
        }
        
        self.decoder_state2embedding: MLP
        self.strategy: BidirTrainStrategyBase | None = strategy
        self.transductive_test = False

    def forward(self, x):
        raise NotImplemented
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler_config
        }
    
    def training_step(self, batch, _batch_idx):
        assert self.strategy is not None
        return self.strategy.training_step(self, batch, _batch_idx)
        
    def on_train_epoch_end(self):
        assert self.strategy is not None
        return self.strategy.on_train_epoch_end(self)
        
    def validation_step(self, batch, _batch_idx):
        assert self.strategy is not None
        return self.strategy.validation_step(self, batch, _batch_idx)

    def on_validation_epoch_end(self):
        assert self.strategy is not None
        return self.strategy.on_validation_epoch_end(self)

    def test_step(self, batch, _batch_idx):
        assert self.strategy is not None
        return self.strategy.test_step(self, batch, _batch_idx)

    def on_test_epoch_end(self):
        assert self.strategy is not None
        return self.strategy.on_test_epoch_end(self)


# two-way reconstruction model, essentially putting together d2p and p2d and bridging them with embedding prediction
class biDirReconModelRNN(biDirReconModelBase):
    def __init__(self,
                            
        d2p_num_encoder_layers: int,
        d2p_dropout_p: float,
        d2p_use_vae_latent: bool,
        d2p_inference_decode_max_length: int,
        d2p_use_bidirectional_encoder: bool,
        d2p_decode_mode: str, # 'greedy_search' | 'beam_search'
        d2p_beam_search_alpha: float | None,
        d2p_beam_size: int | None,
        d2p_lang_embedding_when_decoder: bool,
        d2p_feedforward_dim: int,
        d2p_embedding_dim: int | None,
        d2p_model_size: int,

        p2d_num_encoder_layers: int,
        p2d_dropout_p: float,
        p2d_use_vae_latent: bool,
        p2d_inference_decode_max_length: int,
        p2d_use_bidirectional_encoder: bool,
        p2d_decode_mode: str, # 'greedy_search' | 'beam_search'
        p2d_beam_search_alpha: float | None,
        p2d_beam_size: int | None,
        p2d_lang_embedding_when_decoder: bool,
        p2d_prompt_mlp_with_one_hot_lang: bool,
        p2d_gated_mlp_by_target_lang: bool,
        p2d_feedforward_dim: int,
        p2d_embedding_dim: int | None,
        p2d_model_size: int,
        p2d_all_lang_summary_only: bool,
        
        **kwargs # passed to base class
    ) -> None:
        super().__init__(**kwargs)
        # self.save_hyperparameters()
        
        if self.universal_embedding:
            self.embeddings = Embedding(
                embedding_dim=self.universal_embedding_dim,
                num_ipa_tokens=len(self.ipa_vocab),
                num_langs=len(self.lang_vocab)
            ).to(self.device)
            d2p_embedding_dim = self.universal_embedding_dim
            p2d_embedding_dim = self.universal_embedding_dim

        self.d2p: Seq2SeqRNN = Seq2SeqRNN( # daughter to proto
            ipa_vocab = self.ipa_vocab,
            lang_vocab = self.lang_vocab,
            
            num_encoder_layers = d2p_num_encoder_layers,
            dropout_p = d2p_dropout_p,
            feedforward_dim = d2p_feedforward_dim,
            embedding_dim = d2p_embedding_dim,
            model_size = d2p_model_size,
            inference_decode_max_length = d2p_inference_decode_max_length,
            use_vae_latent = d2p_use_vae_latent,
            use_bidirectional_encoder = d2p_use_bidirectional_encoder,
            init_embedding = self.embeddings,
            training_mode = 'encodeWithLangEmbedding',
            logger_prefix = 'd2p',
            lr = self.lr,
            decode_mode = d2p_decode_mode,
            beam_search_alpha = d2p_beam_search_alpha,
            beam_size = d2p_beam_size,
            lang_embedding_when_decoder = d2p_lang_embedding_when_decoder,
            
            use_xavier_init=self.use_xavier_init,
            max_epochs = self.max_epochs,
            warmup_epochs = self.warmup_epochs,
            beta1 = self.beta1,
            beta2 = self.beta2,
            eps = self.eps,            
            
            encoder_takes_prev_decoder_out=False,
            prompt_mlp_with_one_hot_lang=False,
            gated_mlp_by_target_lang=False,
            all_lang_summary_only = True,

        )
        
        self.p2d: None | Seq2SeqRNN = None if (not self.has_p2d) else Seq2SeqRNN( # proto to daughter
            ipa_vocab = self.ipa_vocab,
            lang_vocab = self.lang_vocab, 
            num_encoder_layers = p2d_num_encoder_layers,
            dropout_p = p2d_dropout_p,
            feedforward_dim = p2d_feedforward_dim,
            embedding_dim = p2d_embedding_dim,
            model_size = p2d_model_size,
            inference_decode_max_length = p2d_inference_decode_max_length,
            use_vae_latent = p2d_use_vae_latent,
            use_bidirectional_encoder = p2d_use_bidirectional_encoder,
            init_embedding = self.embeddings,
            training_mode= 'encodeTokenSeqAppendTargetLangToken',
            logger_prefix = 'p2d',
            lr = self.lr,
            decode_mode = p2d_decode_mode,
            beam_search_alpha = p2d_beam_search_alpha,
            beam_size = p2d_beam_size,
            lang_embedding_when_decoder = p2d_lang_embedding_when_decoder,
            prompt_mlp_with_one_hot_lang=p2d_prompt_mlp_with_one_hot_lang,
            gated_mlp_by_target_lang=p2d_gated_mlp_by_target_lang,
            
            use_xavier_init=self.use_xavier_init,
            max_epochs = self.max_epochs,
            warmup_epochs = self.warmup_epochs,
            beta1 = self.beta1,
            beta2 = self.beta2,
            eps = self.eps,            
            
            encoder_takes_prev_decoder_out=False,
            all_lang_summary_only = p2d_all_lang_summary_only,

        )
                
        if self.strategy is not None:
            self.strategy.extra_init(self)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.lr,
            betas = (self.beta1, self.beta2),
            eps = self.eps,
            weight_decay=self.weight_decay
        )
        self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            self.warmup_epochs,
            self.max_epochs,
            lr_end=0.000001
        )
        self.scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

# two-way reconstruction model, essentially putting together d2p and p2d and bridging them with embedding prediction
class biDirReconModelTrans(biDirReconModelBase):
    def __init__(self,
                            
        d2p_num_encoder_layers: int,
        d2p_num_decoder_layers: int,
        d2p_nhead: int,
        d2p_dropout_p: float,
        d2p_inference_decode_max_length: int,
        d2p_max_len: int, # input max length
        d2p_feedforward_dim: int,
        d2p_embedding_dim: int | None,

        p2d_num_encoder_layers: int,
        p2d_num_decoder_layers: int,
        p2d_nhead: int,
        p2d_dropout_p: float,
        p2d_inference_decode_max_length: int,
        p2d_max_len: int, # input max length
        p2d_feedforward_dim: int,
        p2d_embedding_dim: int | None,
        p2d_all_lang_summary_only: bool,
        
        **kwargs # passed to base class
    ) -> None:
        super().__init__(**kwargs)
        # self.save_hyperparameters()
        
        if self.universal_embedding:
            self.ipa_embedding = TokenEmbedding(len(self.ipa_vocab), self.universal_embedding_dim).to(self.device)
            self.lang_embedding = TokenEmbedding(len(self.lang_vocab), self.universal_embedding_dim).to(self.device)
            d2p_embedding_dim = self.universal_embedding_dim
            p2d_embedding_dim = self.universal_embedding_dim
        else:
            self.ipa_embedding = None
            self.lang_embedding = None

        self.d2p: EncoderDecoderTransformer = EncoderDecoderTransformer(
            ipa_vocab = self.ipa_vocab,
            lang_vocab = self.lang_vocab,
            num_encoder_layers = d2p_num_encoder_layers,
            num_decoder_layers = d2p_num_decoder_layers,
            embedding_dim = d2p_embedding_dim,
            nhead = d2p_nhead,
            feedforward_dim = d2p_feedforward_dim,
            dropout_p = d2p_dropout_p,
            max_len = d2p_max_len,
            logger_prefix = 'd2p',
            task = 'd2p',
            inference_decode_max_length = d2p_inference_decode_max_length,
            all_lang_summary_only = True,
            
            init_ipa_embedding=self.ipa_embedding,
            init_lang_embedding=self.lang_embedding,
            
            use_xavier_init = self.use_xavier_init,
            lr = self.lr,
            warmup_epochs = self.warmup_epochs,
            max_epochs = self.max_epochs,
            weight_decay = self.weight_decay,

            beta1 = self.beta1,
            beta2 = self.beta2,
            eps = self.eps,
        )
        
        self.p2d: None | EncoderDecoderTransformer = None if (not self.has_p2d) else EncoderDecoderTransformer(
            ipa_vocab = self.ipa_vocab,
            lang_vocab = self.lang_vocab,
            num_encoder_layers = p2d_num_encoder_layers,
            num_decoder_layers = p2d_num_decoder_layers,
            embedding_dim = p2d_embedding_dim,
            nhead = p2d_nhead,
            feedforward_dim = p2d_feedforward_dim,
            dropout_p = p2d_dropout_p,
            max_len = p2d_max_len,
            logger_prefix = 'p2d',
            task = 'p2d',
            inference_decode_max_length = p2d_inference_decode_max_length,
            all_lang_summary_only = p2d_all_lang_summary_only,

            init_ipa_embedding=self.ipa_embedding,
            init_lang_embedding=self.lang_embedding,

            use_xavier_init = self.use_xavier_init,
            lr = self.lr,
            warmup_epochs = self.warmup_epochs,
            max_epochs = self.max_epochs,
            weight_decay = self.weight_decay,
            
            beta1 = self.beta1,
            beta2 = self.beta2,
            eps = self.eps,
        )
                
        if self.strategy is not None:
            self.strategy.extra_init(self)
            
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.lr,
            betas = (self.beta1, self.beta2),
            eps = self.eps,
            weight_decay=self.weight_decay
        )
        self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            self.warmup_epochs,
            self.max_epochs,
            lr_end=0.000001
        )
        self.scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
