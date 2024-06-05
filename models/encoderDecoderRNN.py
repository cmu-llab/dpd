from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.rescoring import Reranker, Rescorer
    from lib.rescoring import BatchedGreedyBasedRerankerBase, BatchedReranker, BatchedRescorer

import random
import models.utils as utils
import operator
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, unpack_sequence
from torch import Tensor
from tqdm import tqdm
from lib.vocab import Vocab
from pytorch_lightning.loggers import WandbLogger
import wandb
from einops import rearrange, reduce, repeat
from queue import PriorityQueue
import numpy as np
from .partials.gatedmlp import GatedMLP, GatedMLPConcat
from lib.tensor_utils import collate_endnodes_to_tensors, padded_stack, sequences_equal, sort_by_permutation_3d

from specialtokens import *
from prelude import *

from .partials.embedding import Embedding
from .partials.mlp import MLP
from .partials.attention import Attention
from .partials.rnn import GRURNN
import transformers

DBG = False

def dbg(*args, **kwargs):
    if DBG:
        print(*args, **kwargs)

def dbgassert(cond):
    if DBG:
        assert cond

class Seq2SeqRNN(pl.LightningModule):
    def __init__(self,
            ipa_vocab: Vocab,
            lang_vocab: Vocab,
            num_encoder_layers: int,
            dropout_p: float,
            feedforward_dim: int,
            embedding_dim: int,
            model_size: int,
            
            inference_decode_max_length: int, # e.g. 15
            use_xavier_init: bool ,
            lr: float, # initial lr, e.g. 0.0001
            warmup_epochs: int,
            max_epochs: int,
            
            use_vae_latent: bool,
            use_bidirectional_encoder: bool,
            encoder_takes_prev_decoder_out: bool, # enable for second vae if passing RNN hidden instead of embedding
            init_embedding: Embedding | None, # whether to initialise with another embedding
            training_mode: str, # 'encodeWithLangEmbedding' | 'encodeTokenSeqAppendTargetLangToken'
                # encodeWithLangEmbedding - Input is ipa tokens and lang tokens, Output is ipa tokens
                # encodeTokenSeqAppendTargetLangToken - Input is ipa tokens with target language after <eos>, Output is ipa tokens in the designated target language
            
            logger_prefix: str, # e.g. 'p2d'
            all_lang_summary_only: bool, # true to disable per language eval
            decode_mode: str, # 'greedy_search' | 'beam_search'
            beam_search_alpha: float | None, # e.g. 1.0
            beam_size: int | None, # e.g. 5
            lang_embedding_when_decoder: bool,
            prompt_mlp_with_one_hot_lang: bool, # only for 'encodeTokenSeqAppendTargetLangToken' mode
            gated_mlp_by_target_lang: bool, # only for 'encodeTokenSeqAppendTargetLangToken' mode
            
            beta1: float,
            beta2: float,
            eps: float,
        ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        
        self.ipa_vocab = ipa_vocab
        self.lang_vocab = lang_vocab
        self.protolang: str = lang_vocab.protolang # string like 'Middle Chinese (Baxter and Sagart 2014)'
        self.logger_prefix = logger_prefix
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.all_lang_summary_only = all_lang_summary_only
        
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_encoder_layers # they have to be the same for final hidden to pass from encoder to decoder
        self.use_bidirectional_encoder = use_bidirectional_encoder
        D = (1,2)[self.use_bidirectional_encoder] # 1 if unidirectional, 2 if bidirectional
        self.D = D
        
        self.model_size = model_size

        self.inference_decode_max_length = inference_decode_max_length
        self.training_mode = training_mode
        self.lang_embedding_when_decoder = lang_embedding_when_decoder # TODO should be called lang_embedding_when_decoding...
        
        # assert not (prompt_mlp_with_one_hot_lang and gated_mlp_by_target_lang)
        self.prompt_mlp_with_one_hot_lang = prompt_mlp_with_one_hot_lang
        self.gated_mlp_by_target_lang = gated_mlp_by_target_lang
        
        self.num_possible_decode_target_langs = len(self.lang_vocab.daughter_langs)
        self.possible_target_lang_langs = self.lang_vocab.to_indices(self.lang_vocab.daughter_langs) # list of ints, indices in lang_vocab.
        self.min_possible_target_lang_langs_idx = min(self.possible_target_lang_langs)
        
        
        self.decode_mode = decode_mode
        self.beam_search_alpha = beam_search_alpha
        self.beam_size = beam_size
        assert self.decode_mode in ('greedy_search', 'beam_search')
        if self.decode_mode == 'beam_search':
            assert self.beam_search_alpha is not None
            assert self.beam_size is not None
        
        # shared embedding across all languages
        self.embeddings: Embedding
        match init_embedding:
            case None:
                self.embeddings = Embedding(
                    embedding_dim=embedding_dim,
                    num_ipa_tokens=len(self.ipa_vocab),
                    num_langs=len(self.lang_vocab)
                )
            case _:
                self.embeddings = init_embedding

        self.dropout = nn.Dropout(dropout_p)
        
        self.encoder_rnn = GRURNN(
            input_size = (embedding_dim if not encoder_takes_prev_decoder_out else model_size),
            hidden_size = model_size,
            num_layers = num_encoder_layers,
            bidirectional = use_bidirectional_encoder,
        )
        self.bidirectional_to_unidirectional_bridge = nn.Linear(model_size * D, model_size) if use_bidirectional_encoder else None
        self.decoder_rnn = GRURNN(
            input_size = embedding_dim + model_size * D,
            hidden_size = model_size * D,
            num_layers = self.num_decoder_layers,
        )

        self.V = len(ipa_vocab) # vocab size
        match self.gated_mlp_by_target_lang:
            case True:
                self.mlp = GatedMLPConcat(
                    input_dim = model_size * D + (self.num_possible_decode_target_langs if self.prompt_mlp_with_one_hot_lang else 0),
                    feedforward_dim = feedforward_dim,
                    gate_count=self.num_possible_decode_target_langs,
                    output_size = self.V,
                )
            case False:
                self.mlp = MLP(
                    input_dim = model_size * D + (self.num_possible_decode_target_langs if self.prompt_mlp_with_one_hot_lang else 0),
                    feedforward_dim = feedforward_dim, 
                    output_size = self.V
                )
        self.attention = Attention(
            hidden_dim = model_size * D,
            embedding_dim = embedding_dim
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        
        self.use_vae_latent = use_vae_latent
        if self.use_vae_latent:
            self.to_mu = nn.Linear(
                D * model_size, 
                D * model_size, 
            )
            self.to_logvar = nn.Linear(
                D * model_size, 
                D * model_size, 
            )
        
        # Xavier initialization
        self.use_xavier_init = use_xavier_init
        if self.use_xavier_init:
            print("performing Xavier initialization")
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            # remember to reset PAD embedding to 0
            self.embeddings.set_pad_embedding_to_zero()
            
        self.eval_step_outputs = []
        self.evaled_on_target_langs = set() # keep track of which target langs we've evaled on if we're using encodeTokenSeqAppendTargetLangToken
        self.eval_samples_table = None
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = lr,
            betas = (beta1, beta2), # default
            eps = eps, # default
        )
        self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            self.warmup_epochs,
            self.max_epochs,
            lr_end=0.000001
        )
        self.scheduler_config = {
            "scheduler": self.scheduler,
            # "monitor": f'{self.logger_prefix}/val/accuracy',
            "interval": "epoch",
            "frequency": 1,
        }
        
        self.transductive_test = False
        self.enable_logging = True # TODO remove. This is for historical reason. Use try except instead.

    def get_one_hot_target_lang(self, target_lang_langs: Tensor) -> Tensor:
        assert target_lang_langs != None
        assert self.prompt_mlp_with_one_hot_lang
        return F.one_hot(target_lang_langs - self.min_possible_target_lang_langs_idx, num_classes=self.num_possible_decode_target_langs).float().to(self.device)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler_config
        }
    
    def encode(self, source_tokens: Tensor, source_langs: Tensor | None, source_seqs_lens: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        # INPUT:
        # source_tokens: (N, L); L is seq len
        # source_langs: (N, L) or None if no lang embedding
        # source_seqs_lens: (N, 1) length of respective unpaded sequences

        embedded_cognateset: Tensor = self.embeddings(source_tokens, source_langs)
        embedded_cognateset: Tensor = self.dropout(embedded_cognateset) # (N, L, embedding dim)

        packed_sequences = pack_padded_sequence(embedded_cognateset, rearrange(source_seqs_lens, 'N 1 -> N').to('cpu'), batch_first=True, enforce_sorted=False)

        (packed_output, h_n) = self.encoder_rnn(packed_sequences)
        # h_n: (num encoder layer * D, N, hidden size); final hidden state
        
        output_list = unpack_sequence(packed_output)
        output = pad_sequence(output_list, batch_first=True, padding_value=0)
        
        if self.use_bidirectional_encoder:
            # manually cat h_n from both directions for every layer
            h_n = torch.cat([h_n[0:h_n.shape[0]:2], h_n[1:h_n.shape[0]:2]], dim=-1)
        
        # RETURNS:
        # output: (N, L, H * D); last layer output of RNN
        # h_n: (num encoder layer, N, H * D); final hidden state
        # embedded_cognateset: (N, L, E)
        return (output, h_n), embedded_cognateset

    def encode_and_prepare_decode(self,                                   
        source_tokens: Tensor, 
        source_langs: Tensor | None,
        source_seqs_lens: Tensor,
        inject_embedding: Tensor | None = None, # replace embedding fed into encoder
    ):
        
        source_padding_mask: Tensor = (source_tokens == PAD_IDX)
        # source_padding_mask: (N, L)
        
        match inject_embedding:
            case None:
                (encoder_states, encoder_h_n), embedded_cognateset = self.encode(source_tokens, source_langs, source_seqs_lens)
            case _: # inject embedding
                (encoder_states, encoder_h_n), embedded_cognateset = self.injected_embedding_encode(inject_embedding)
        
        # encoder_states: (N, L, H * D); last layer output of RNN
        # h_n: (num encoder layer, N, H * D); final hidden state
        # embedded_cognateset: (N, L, E)
        
        memory = rearrange(encoder_h_n[-1], 'N DH -> N 1 DH') # (1, N, H * D)

        # memery: (1, N, H); keeping last layer of h_n
        
        match self.use_vae_latent:
            case True:
                memory, mu, logvar = self.memory_to_z(memory, sampling=True)
            case False:
                memory = memory
                mu, logvar = None, None
                        
        return encoder_states, encoder_h_n, memory, mu, logvar, embedded_cognateset, source_padding_mask

    # injected encode - takes embedding directly, rather than tokens
    def injected_embedding_encode(self, embedded_seq: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:

        embedded_cognateset: Tensor = embedded_seq
        embedded_cognateset: Tensor = self.dropout(embedded_cognateset) # (N, L, embedding dim)

        (output, h_n) = self.encoder_rnn(embedded_cognateset)
        # output: (N, L, H); last layer output of RNN
        # h_n: (D * num encoder layer, N, H); final hidden state
        
        if self.use_bidirectional_encoder:
            # manually cat h_n from both directions for every layer
            h_n = torch.cat([h_n[0:h_n.shape[0]:2], h_n[1:h_n.shape[0]:2]], dim=-1)

        return (output, h_n), embedded_cognateset
    
    # for vae
    def memory_to_z(self, memory, sampling=True) -> tuple[Tensor, Tensor, Tensor]:
        mu: Tensor = self.to_mu(memory)
        # mu: (1, N, H)
        logvar: Tensor = self.to_logvar(memory)
        # logvar: (1, N, H)
        z: Tensor = utils.vae_reparametrise(mu, logvar, sampling=sampling)
        return z, mu, logvar

    def batch_size_check(self, N: int, source_tokens: Tensor, source_langs: Tensor | None, target_tokens: Tensor | None, target_lang_ids: Tensor | None = None):
        match self.training_mode:
            case 'encodeWithLangEmbedding':
                assert(source_langs != None and source_tokens.shape[0] == source_langs.shape[0] == N)
            case 'encodeTokenSeqAppendTargetLangToken':
                assert((target_lang_ids == None or target_lang_ids.shape[0] == N) and source_tokens.shape[0] == N)
        
        assert(target_tokens == None or target_tokens.shape[0] == N)
        
    def decode_once(self,
        tkn_ids: Tensor, # (N, 1)
        lang_ids: Tensor | None, # (N, 1) or None 
        prev_attention_weights: Tensor, # (N, 1, D * H)
        prev_hidden: Tensor, # decoder_state: (num layers, N, H_decoder)
        # the following are from the encoder
        encoder_outputs: Tensor, # (N, Ls, D * H)
        embedded_cognateset: Tensor, # (N, Ls, E)
        source_padding_mask: Tensor # (N, Ls)
    ):
        N = tkn_ids.shape[0]
        assert (tkn_ids.shape == (N, 1))
        assert (lang_ids == None or lang_ids.shape == (N, 1))
        assert (prev_attention_weights.shape == (N, 1, self.model_size * self.D))
        assert (encoder_outputs.shape[0] == N)
        assert (embedded_cognateset.shape[0] == N)
        assert (source_padding_mask.shape[0] == N)
        
        
        seq_embedded = self.embeddings(tkn_ids, lang_ids) # (N, 1, E)
        decoder_input = self.dropout(torch.cat(
            (seq_embedded, prev_attention_weights), 
            dim=-1 # along last dim
        )) # (N, 1, D*H + E)
        
        decoder_output, hidden = self.decoder_rnn(decoder_input, prev_hidden) # (N, 1, D*H), (N, 1, H)
        
        attention_weights: Tensor = self.attention(
            query = decoder_output, # (N, 1, D*H)
            keys = encoder_outputs,  # (N, Ls, D*H)
            encoded_input = embedded_cognateset, # (N, Ls, E)
            padding_mask = source_padding_mask # (N, Ls)
        )
        attention_weights = rearrange(attention_weights, 'N Ls -> N 1 Ls')

        return decoder_output, hidden, attention_weights
        # (N, 1, D * H), (N, 1, H), (N 1 Ls)

    def greedy_decode(self, 
        source_tokens: Tensor, # (N, L)
        source_langs: Tensor | None, # (N, L)
        source_seqs_lens: Tensor, # (N, 1)
        target_langs: Tensor | None, # (N, 1)
    ):
        N = source_tokens.shape[0]
        self.batch_size_check(N, source_tokens, source_langs, None, target_langs)
        
        # === ENCODE ===
        
        encoder_states, encoder_h_n, memory, _mu, _logvar, embedded_cognateset, source_padding_mask = self.encode_and_prepare_decode(source_tokens, source_langs, source_seqs_lens, None)
        
        encoder_states = self.dropout(encoder_states) 
        

        # === INITIAL DECODE ===
                
        decoder_output, hidden, attention_weights, (bos_tkn_vec, sep_lang_vec) = self.initial_decode(N, memory, encoder_h_n, encoder_states, embedded_cognateset, source_padding_mask)
        
        decoder_output = self.dropout(decoder_output)


        # === DECODE SEQUENCE ===

        reconstruction = [bos_tkn_vec]
        reached_eos = torch.zeros((N, 1), dtype=torch.bool).to(self.device)
        i = 0
        
        while i < self.inference_decode_max_length:
            
            # == CLASSIFY ==

            classification_input = decoder_output + attention_weights
            # classification_input: (N, L = 1, E)
                       
            match (self.prompt_mlp_with_one_hot_lang, self.gated_mlp_by_target_lang):
                case (True, True):
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                    tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
                case (True, False):
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                    tkn_scores = self.mlp(classification_input)
                case (False, True):
                    tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
                case (False, False):
                    tkn_scores = self.mlp(classification_input)
            # tkn_scores: (N, 1, V) this is before softmax

            tkn_pred = torch.argmax(tkn_scores.squeeze(dim=1), dim=-1)
            tkn_pred = rearrange(tkn_pred, 'N-> N 1')
            # tkn_pred: (N, 1)
            
            # dealing with eos. reached_eos keeps track of which sequences have reached eos and sets all following tokens to PAD
            tkn_pred = (reached_eos == 1) * PAD_IDX | (reached_eos == 0) * tkn_pred
            reached_eos = reached_eos | (tkn_pred == EOS_IDX)
            
            reconstruction.append(tkn_pred)
            
            # == ONTO NEXT RECURRENCE ==
            
            target_lang_vec = target_langs 

            decoder_output, hidden, attention_weights = self.decode_once(tkn_pred, target_lang_vec if self.lang_embedding_when_decoder else None, attention_weights, hidden, encoder_states, embedded_cognateset, source_padding_mask)
            
            decoder_output = self.dropout(decoder_output)

            # == EXIT IF everything in tkn_pred is EOS ==
            
            if torch.all(reached_eos == 1):
                break
            else:
                i += 1

        # by this point, reconstruction is list with len L_decoded of predictions of shape (N, 1). We want to stack them into (N, L_decoded)
        return torch.cat(reconstruction, dim=1)

    def beam_search_decode(self, 
        source_tokens: Tensor, # (N, L)
        source_langs: Tensor | None, # (N, L) is any
        source_seqs_lens: Tensor, # (N, 1)
        target_langs: Tensor | None, # (N, L) is any
        beam_size: int,
    ):
        N = source_tokens.size(0)
        B = beam_size
        self.batch_size_check(N, source_tokens, source_langs, None)


        # === ENCODE ===
        
        encoder_states, encoder_h_n, memory, _mu, _logvar, embedded_cognateset, source_padding_mask = self.encode_and_prepare_decode(source_tokens, source_langs, source_seqs_lens, None)
        
        encoder_states = self.dropout(encoder_states) 


        # === INITIAL DECODE ===
                
        decoder_output, hidden, attention_weights, (bos_tkn_vec, sep_lang_vec) = self.initial_decode(N, memory, encoder_h_n, encoder_states, embedded_cognateset, source_padding_mask)
        decoder_output = self.dropout(decoder_output)
        
        
        # === BEAM SEARCH === 
        
        ## == set up ==
        
        beam_sequences = rearrange(repeat(bos_tkn_vec, 'N 1 -> N B', B=B), 'N B -> N B 1') # all beams start with BOS
        beam_log_probs = torch.zeros(B * N).to(self.device) # (BN) # TODO should this be 1 or 0?
        lenient_endnodes = [[] for i in range(N)] # one endnode list for each batch item, unfiltered for if the seq has reached eos before
        
        ## == first blow N to BN, filling the beam ==
        
        ### classification and pick tops
        
        classification_input = decoder_output + attention_weights

        match (self.prompt_mlp_with_one_hot_lang, self.gated_mlp_by_target_lang):
            case (True, True):
                classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
            case (True, False):
                classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                tkn_scores = self.mlp(classification_input)
            case (False, True):
                tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
            case (False, False):
                tkn_scores = self.mlp(classification_input)
        # tkn_scores: (N, 1, V) this is before softmax

        log_tkn_probs_ = torch.log_softmax(tkn_scores, dim=-1) # (N, 1, V)
        log_tkn_probs = rearrange(log_tkn_probs_, 'N 1 V -> N V') # (N, V)
        
        log_probs, tkn_preds = torch.topk(log_tkn_probs, B, sorted=True) # (N, B), (N, B)
        dbg('log_probs', log_probs)
        dbg('tkn_preds', tkn_preds)
        
        tkn_preds_entry = rearrange(tkn_preds, 'N B -> N B 1')
                
        beam_sequences = torch.cat((beam_sequences, tkn_preds_entry), dim=-1) # (N, B, 2)
        beam_log_probs = beam_log_probs + rearrange(log_probs, 'N B -> (B N)') # (BN)
        dbg('beam_sequences', beam_sequences)
        dbg('beam_log_probs', beam_log_probs)
        
        beam_log_probs_batch_first = beam_log_probs[(repeat(torch.arange(B), "B -> N B", N=N) * N) + rearrange(torch.arange(N), "N -> N 1")]   # (N, B)
        
        dbg("\n\n")

        ### add endnodes eos
        
        tkn_pred_eos_mask_ = tkn_preds_entry == EOS_IDX # (N, B, 1)
        tkn_pred_eos_mask = rearrange(tkn_pred_eos_mask_, 'N B 1 -> N B') # (N B)
                
        match torch.all(tkn_pred_eos_mask == False).item():
            case True: # nothing reached EOS, no endnodes to add
                pass 
            case False: # some reached EOS, add endnodes, and TODO set the cumulative log prob to -inf to stop further expansion of the branch... turns out problematic as we already selected those sequences for expansion...
                for batch_id in range(N):
                    endnodes_in_batch = beam_sequences[batch_id, tkn_pred_eos_mask[batch_id]]
                    log_probs_for_endnodes_in_batch = beam_log_probs_batch_first[batch_id, tkn_pred_eos_mask[batch_id]]
                    log_probs_for_endnodes_in_batch_items = [log_prob.item() for log_prob in log_probs_for_endnodes_in_batch]
                    lenient_endnodes[batch_id].extend(zip(log_probs_for_endnodes_in_batch_items, endnodes_in_batch))
        
        ### repeat everything from root node B times to get B beams
        attention_weights = repeat(attention_weights, 'N L E -> (B N) L E', B=B)
        hidden = repeat(hidden, 'DNl N H -> DNl (B N) H', B=B) # Nl is num layers
        encoder_states = repeat(encoder_states, 'N L DH -> (B N) L DH', B=B)
        embedded_cognateset = repeat(embedded_cognateset, 'N Le E -> (B N) Le E', B=B)
        source_padding_mask = repeat(source_padding_mask, 'N Le -> (B N) Le', B=B)
        
        reached_eos = torch.zeros(B * N, dtype=torch.bool).to(self.device) # (BN)
        
        ## == next keep pushing forward with the same beam size with BN as batch size ==
        
        for i in range(self.inference_decode_max_length):
        
            ### prep next decoder input
            
            next_tkn_col = rearrange(tkn_preds, 'N B -> (B N) 1') # (BN, 1)
            if target_langs != None:
                target_lang_col = repeat(target_langs, 'N 1 -> (B N) 1', B=B) # (BN, 1)
            else:
                target_lang_col = None
            dbg('next_tkn_col', next_tkn_col)
            
            ### decode
            
            decoder_output, hidden, attention_weights = self.decode_once(next_tkn_col, target_lang_col if self.lang_embedding_when_decoder else None, attention_weights, hidden, encoder_states, embedded_cognateset, source_padding_mask)
            decoder_output = self.dropout(decoder_output)
            
            ### classify and pick cumulative tops
            
            classification_input = decoder_output + attention_weights

            match (self.prompt_mlp_with_one_hot_lang, self.gated_mlp_by_target_lang):
                case (True, True):
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                    tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
                case (True, False):
                    dbg('target_lang_col', target_lang_col)
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_lang_col)), dim=-1)
                    tkn_scores = self.mlp(classification_input)
                case (False, True):
                    dbg('target_lang_col', target_lang_col)
                    tkn_scores = self.mlp(rearrange(classification_input, 'BN 1 E -> BN E'), target_lang_col - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'BN V -> BN 1 V')
                case (False, False):
                    tkn_scores = self.mlp(classification_input)
            # tkn_scores: (BN, 1, V) this is before softmax

            log_tkn_probs_ = torch.log_softmax(tkn_scores, dim=-1) # (BN, 1, V)
            log_tkn_probs = rearrange(log_tkn_probs_, 'N 1 V -> N V') # (BN, V)
            
            cumulative_log_probs_ = repeat(beam_log_probs, 'BN -> BN V', V=self.V) # (BN, V)
            dbg('cumulative_log_probs_', cumulative_log_probs_)
            dbg('log_tkn_probs', log_tkn_probs)
            cumulative_log_probs = cumulative_log_probs_ + log_tkn_probs # (BN, V)
            dbg('cumulative_log_probs', cumulative_log_probs)
            # dbg('cumulative_log_probs', cumulative_log_probs)
            
            cumulative_log_probs_batch_first_ = cumulative_log_probs[(repeat(torch.arange(B), "B -> N B", N=N) * N) + rearrange(torch.arange(N), "N -> N 1"), :]   # (N, B, V)
            cumulative_log_probs_batch_first = rearrange(cumulative_log_probs_batch_first_, 'N B V -> N (B V)') # (N, BV)

            log_probs, indexes = torch.topk(cumulative_log_probs_batch_first, B, sorted=True) # (N, B), (N, B)
            from_beams = indexes // self.V         # (N, B) which beam each top came from
            tkn_preds = indexes % self.V           # (N, B) but in range of vocab
            dbg('indexes', indexes)
            dbg('log_probs', log_probs)
            dbg('tkn_preds', tkn_preds)
            dbg('from_beams', from_beams)
            
            ### grabbing the right hidden and attention weights for the beams
            
            beam_ids = rearrange(from_beams, 'N B -> (B N)') # (BN)
            hidden = hidden[:, (torch.arange(N * B).to(self.device) % N) + (beam_ids * N), :]                         # selecting the hidden corresponding to the training example and beam (DNl, BN, H)
            attention_weights = attention_weights[(torch.arange(N * B) % N).to(self.device) + (beam_ids * N), :, :]   # (BN, L, E)
            
            ### growing the tree
            
            tkn_preds_entry = rearrange(tkn_preds, 'N B -> N B 1')
            dbg('tkn_preds_entry', tkn_preds_entry)
            
            selected_sequences = beam_sequences[torch.arange(N).to(self.device).unsqueeze(1), from_beams, :] # (N, B, l)
            dbg(beam_sequences)
            dbg(selected_sequences)
            selected_sequences_log_probs = beam_log_probs[(from_beams.T.flatten() * N) + ((torch.arange(N * B).to(self.device)) % N)] # (BN)
            reached_eos = reached_eos[(from_beams.T.flatten() * N) + ((torch.arange(N * B).to(self.device)) % N)] # (BN)
            
            beam_sequences = torch.cat((selected_sequences, tkn_preds_entry), dim=-1) # (N, B, l++)
            beam_log_probs = 00000000000 + rearrange(log_probs, 'N B -> (B N)') # (BN) 
            dbg('beam_sequences', beam_sequences)
            dbg('beam_log_probs', beam_log_probs)        
            
            beam_log_probs_batch_first = beam_log_probs[(repeat(torch.arange(B).to(self.device), "B -> N B", N=N) * N) + rearrange(torch.arange(N).to(self.device), "N -> N 1")]   # (N, B)

            ### add endnodes eos
        
            tkn_pred_eos_mask_ = tkn_preds_entry == EOS_IDX # (N, B, 1)
            tkn_pred_eos_mask = rearrange(tkn_pred_eos_mask_, 'N B 1 -> N B') # (N B)
            
            reached_eos = reached_eos | rearrange(tkn_pred_eos_mask, 'N B -> (B N)') 
                    
            match torch.all(tkn_pred_eos_mask == False).item():
                case True: # nothing reached EOS, no endnodes to add
                    pass 
                case False: # some reached EOS, add endnodes, and TODO set the cumulative log prob to -inf to stop further expansion of the branch... turns out problematic as we already selected those sequences for expansion...
                    for batch_id in range(N):
                        endnodes_in_batch = beam_sequences[batch_id, tkn_pred_eos_mask[batch_id]]
                        log_probs_for_endnodes_in_batch = beam_log_probs_batch_first[batch_id, tkn_pred_eos_mask[batch_id]]
                        log_probs_for_endnodes_in_batch_items = [log_prob.item() for log_prob in log_probs_for_endnodes_in_batch]
                        lenient_endnodes[batch_id].extend(zip(log_probs_for_endnodes_in_batch_items, endnodes_in_batch))

            match torch.all(reached_eos == True).item():
                case True: # all reached EOS, no need to continue. note this doesn't catch the case where all beams hit EOS, because they could have hit EOS at different times
                    break
                case False: # continue beam search
                    pass

            dbg('\n\n')

        # === ENDNODE SELECTION AND RERANKING ===
        processed_endnodes = [[] for _ in range(N)]
        best_sequences_acc = []
        for batch_id in range(N):
            lenient_endnodes_for_batch = lenient_endnodes[batch_id]
            for log_prob, endnode_seq in lenient_endnodes_for_batch:
                if torch.sum(endnode_seq == EOS_IDX) > 1:
                    pass # ignore endnodes with multiple EOS
                elif len(endnode_seq) <= 2:
                    pass # ignore endnodes with only BOS and EOS
                else:
                    normalized_log_prob = log_prob / ((float(len(endnode_seq) - 1)) ** self.beam_search_alpha)
                    processed_endnodes[batch_id].append({
                        'endnode_seq': endnode_seq,
                        'normalized_log_prob': normalized_log_prob, 
                        'log_prob': log_prob, 
                    })
                    
            processed_endnodes[batch_id] = sorted(processed_endnodes[batch_id], key=operator.itemgetter('normalized_log_prob'), reverse=True)
            if len(processed_endnodes[batch_id]) > 0:
                best_sequences_acc.append(processed_endnodes[batch_id][0]['endnode_seq'])
            else:
                best_sequences_acc.append(beam_sequences[batch_id, 0, :]) # append something that hasn't reached eos from top of beam, so that we don't crash
        best_seqs_padded = pad_sequence(best_sequences_acc, batch_first=True, padding_value=PAD_IDX)

        return beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded
    
    def batched_rerank_endnodes(self, 
        beam_sequences: Tensor, # beam sequences from beam search for fallback purpose
        processed_endnodes: list[dict], # list[N] of lists[Nc] of dicts from beam search, Nc is num candidates
        num_candidates: int, # max number of candidates to consider for each endnode
        
        reranker: BatchedReranker, # function that looks at an endnode seq and some target seqs and returns a score for the endnode seq
        rescorer: BatchedRescorer, # function that looks at reranker output and original log prob and returns some adjusted score
        
        **reranker_kw_args,
    ): # TODO refactor out
        
        daughter_seqs_s = reranker_kw_args['daughter_seqs_s'] # (N, Nd, Ld)
        target_ipa_langs_s = reranker_kw_args['target_ipa_langs_s'] # (N, Nd, 1)
        target_lang_langs_s = reranker_kw_args['target_lang_langs_s'] # (N, Nd, 1)
        valid_daughters_mask = (target_ipa_langs_s != PAD_IDX) # (N, Nd, 1)
        
        N = len(processed_endnodes)
        Nd = valid_daughters_mask.shape[1]
        Nc = num_candidates # how many candidates to consider

        collated_candidates, collated_normalized_log_prob, collated_log_prob = collate_endnodes_to_tensors(N, Nc, Nd, processed_endnodes, self.device) # (N, Nc, Lc), (N, Nc, 1), (N, Nc, 1)
        
        broadcasted_collated_candidates = repeat(collated_candidates, 'N Nc Lc -> N Nc Nd Lc', Nd=Nd) # (N, Nc, Nd, Lc)
        broadcasted_valid_daughters_mask = repeat(valid_daughters_mask, 'N Nd 1 -> N Nc Nd 1', Nc=Nc) # (N, Nc, Nd, 1)
        broadcasted_daughter_seqs_s = repeat(daughter_seqs_s, 'N Nd Ld -> N Nc Nd Ld', Nc=Nc) # (N, Nc, Nd, Ld)


        reranker_scores = reranker(N, Nc, Nd, target_ipa_langs_s, target_lang_langs_s, broadcasted_collated_candidates, broadcasted_valid_daughters_mask, broadcasted_daughter_seqs_s)
        adjusted_scores = rescorer(collated_normalized_log_prob, reranker_scores) # (N, Nc, 1)

        
        rescored_endnodes: rescored_endnodes_t = (
            collated_candidates, # (N, Nc, Lc)
            reranker_scores, # (N, Nc, 1)
            adjusted_scores, # (N, Nc, 1)
            collated_normalized_log_prob, # (N, Nc, 1)
            collated_log_prob, # (N, Nc, 1)
        )
        
        _, sorted_indices = torch.sort(adjusted_scores, dim=-2, descending=True) # (N, Nc, 1)
        sorted_candidates = sort_by_permutation_3d(collated_candidates, sorted_indices) # (N, Nc, Lc)
        best_seqs_padded = sorted_candidates[:, 0, :] # (N, Lc)
        
        return rescored_endnodes, best_seqs_padded
    
        # region LEGACY: rescored_endnodes like beam search API

        # rebuild endnodes
        rescored_endnodes = [[] for _ in range(N)]
        best_sequences_acc = []
        
        for n in range(N):
            for c in range(Nc):
                rescored_endnodes[n].append({
                    'endnode_seq': collated_candidate[n, c, 0],
                    'adjusted_score': adjusted_scores[n, c].item(),
                    'reranker_score': reranker_scores[n, c].item(),
                    'normalized_log_prob': collated_normalized_log_prob[n, c].item(),
                    'log_prob': collated_log_prob[n, c].item(),
                })
            
            rescored_endnodes[n] = sorted(rescored_endnodes[n], key=operator.itemgetter('adjusted_score'), reverse=True)
            if len(rescored_endnodes[n]) > 0:
                best_sequences_acc.append(rescored_endnodes[n][0]['endnode_seq'])
            else:
                best_sequences_acc.append(beam_sequences[n, 0, :]) # append something that hasn't reached eos from top of beam, so that we don't crash
        
        best_seqs_padded = pad_sequence(best_sequences_acc, batch_first=True, padding_value=PAD_IDX)


        # returns a list of lists of dicts, where each dict has keys 'endnode_seq', 'adjusted_score' and each batch is sorted by adjusted_score.
        # also returns best_seqs_padded based on the best adjusted_score for each batch
        return rescored_endnodes, best_seqs_padded
    
        # endregion


    def rerank_endnodes(self, 
        beam_sequences: Tensor, # beam sequences from beam search for fallback purpose
        processed_endnodes: list[dict], # list[N] of lists[Nc] of dicts from beam search, Nc is num candidates
        
        reranker: Reranker, # function that looks at an endnode seq and some target seqs and returns a score for the endnode seq
        rescorer: Rescorer, # function that looks at reranker output and original log prob and returns some adjusted score
        
        **reranker_kw_args,
    ):
        N = len(processed_endnodes)
        rescored_endnodes = [[] for _ in range(N)]
        best_sequences_acc = []
        
        truncate_thresh = 9999 # TODO make hyperparam?
        
        for batch_id in range(N):
            
            # sometimes processed endnodes has more than beam size, so we truncate?
            processed_endnodes_for_batch = processed_endnodes[batch_id]
            if len(processed_endnodes_for_batch) > truncate_thresh:
                processed_endnodes_for_batch = processed_endnodes_for_batch[:truncate_thresh]
            
            for entry in processed_endnodes_for_batch:
                endnode_seq = entry['endnode_seq']
                normalized_log_prob = entry['normalized_log_prob']
                log_prob = entry['log_prob']
                
                reranker_score = reranker(endnode_seq, **reranker_kw_args)
                
                adjusted_score = rescorer(normalized_log_prob, reranker_score)
                
                rescored_endnodes[batch_id].append({
                    'endnode_seq': endnode_seq,
                    'adjusted_score': adjusted_score,
                    'reranker_score': reranker_score,
                    'normalized_log_prob': normalized_log_prob,
                    'log_prob': log_prob,
                })
                
            rescored_endnodes[batch_id] = sorted(rescored_endnodes[batch_id], key=operator.itemgetter('adjusted_score'), reverse=True)
            if len(rescored_endnodes[batch_id]) > 0:
                best_sequences_acc.append(rescored_endnodes[batch_id][0]['endnode_seq'])
            else:
                best_sequences_acc.append(beam_sequences[batch_id, 0, :]) # append something that hasn't reached eos from top of beam, so that we don't crash
        best_seqs_padded = pad_sequence(best_sequences_acc, batch_first=True, padding_value=PAD_IDX)
        
        # returns a list of lists of dicts, where each dict has keys 'endnode_seq', 'adjusted_score' and each batch is sorted by adjusted_score.
        # also returns best_seqs_padded based on the best adjusted_score for each batch
        return rescored_endnodes, best_seqs_padded
    
    # NOTE - this requires the dataloader to combine the d2p and p2d dataloaders
    def reranked_beam_search_decode(self,
        pass1_source_tokens: Tensor, # (N, L_s)
        pass1_source_langs: Tensor | None, # (N, L_s)
        pass1_source_seqs_lens: Tensor, # (N, 1)
        pass1_target_lang_langs: Tensor | None, # (N, L_s)
        pass1_beam_size: int, 
        
        pass2_target_seqs: Tensor, # (N, L_t)
        pass2_target_ipa_langs: Tensor, # (N, L_t)
        pass2_target_lang_langs: Tensor, # (N, L_t)
        
        reranker: Reranker,
        rescorer: Rescorer,
    ): 
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = self.beam_search_decode(pass1_source_tokens, pass1_source_langs, pass1_source_seqs_lens, pass1_target_lang_langs, pass1_beam_size)
                
        rescored_endnodes, best_seqs_padded = self.rerank_endnodes(
            beam_sequences, 
            processed_endnodes, 
            reranker = reranker, 
            rescorer = rescorer,
            
            daughter_seqs = pass2_target_seqs,
            target_ipa_langs = pass2_target_ipa_langs,
            target_lang_langs = pass2_target_lang_langs,
        )
        
        return rescored_endnodes, best_seqs_padded
    
    # NOTE - this requires the dataloader to combine the d2p and p2d dataloaders
    def batched_reranked_beam_search_decode(self,
        pass1_source_tokens: Tensor, # (N, L_s)
        pass1_source_langs: Tensor | None, # (N, L_s)
        pass1_source_seqs_lens: Tensor, # (N, 1)
        pass1_target_lang_langs: Tensor | None, # (N, L_s)
        pass1_beam_size: int, 
        
        pass2_target_seqs_s: Tensor, # (N, Nd, L_t)
        pass2_target_ipa_langs_s: Tensor, # (N, Nd, L_t)
        pass2_target_lang_langs_s: Tensor, # (N, Nd, L_t)
        
        reranker: Reranker,
        rescorer: Rescorer,
    ): 
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = self.beam_search_decode(pass1_source_tokens, pass1_source_langs, pass1_source_seqs_lens, pass1_target_lang_langs, pass1_beam_size)
        
        rescored_endnodes, best_seqs_padded = self.batched_rerank_endnodes(
            beam_sequences, 
            processed_endnodes, 
            reranker = reranker, 
            rescorer = rescorer,
            num_candidates=pass1_beam_size,
            
            daughter_seqs_s = pass2_target_seqs_s,
            target_ipa_langs_s = pass2_target_ipa_langs_s,
            target_lang_langs_s = pass2_target_lang_langs_s,
        )
        
        return rescored_endnodes, best_seqs_padded

    def initial_decode(self,
        N: int,
        memory: Tensor,
        encoder_h_n: Tensor,
        encoder_states: Tensor,
        embedded_cognateset: Tensor,
        source_padding_mask: Tensor,
    ):
        # the initial columns of BOS tokens and corresponding sep lang tokens
        bos_tkn_vec = repeat(torch.LongTensor([BOS_IDX]).to(self.device), '1 -> N 1', N=N)
        sep_lang_vec = repeat(torch.LongTensor([self.lang_vocab.get_idx(SEPARATOR_LANG)]).to(self.device), '1 -> N 1', N=N) 
                
        attention_weights = memory
        # attention_weights: (N, L = 1, H * D); weighted state is last layer of encoder_h_n to begin with, or with noise added if VAE
        
        # if self.use_bidirectional_encoder:
        #     assert self.bidirectional_to_unidirectional_bridge is not None
        #     prev_hidden = self.bidirectional_to_unidirectional_bridge(encoder_h_n)
        # else:
        prev_hidden = encoder_h_n
        # prev_hidden is (Nlayer, N, H)
        
        decoder_output, hidden, attention_weights = self.decode_once(
            bos_tkn_vec, 
            sep_lang_vec if self.lang_embedding_when_decoder else None, 
            attention_weights, # used as prev hidden
            prev_hidden, 
            encoder_states, 
            embedded_cognateset, 
            source_padding_mask
        )
        
        return decoder_output, hidden, attention_weights, (bos_tkn_vec, sep_lang_vec)

 
    # TODO teacher forcing ratio
    # 100% teacher forcing enabled, only used for training, returns loss
    # returns in dictionary form
    # returns logits if plain
    # returns also returns mu and logvar if vae
    # returns decoder_state if get_decoder_states turned on
    def teacher_forcing_decode(self, 
        source_tokens: Tensor, # (N, Ls)
        source_langs: Tensor | None, # (N, Ls)
        source_seqs_lens: Tensor, # (N, 1)
        target_tokens: Tensor, # (N, Lt)
        target_langs: Tensor | None, # (N, 1) used for language embeding; should be lang lang rather than ipa lang
        inject_embedding: Tensor | None = None, # replace embedding fed into encoder
    ) -> dict:
        N = source_tokens.size(0) # viz. batch size
        
        self.batch_size_check(N, source_tokens, source_langs, target_tokens)
        # source_tokens, source_langs: (N, L)
        # target_tokens: (N, L_target)
        
        # N: batch size
        # L: seq len
        # H: hidden size; note H is doubled if bidirectional. still written H here for future tracking
        # E: embedding dim
                
        
        # === ENCODE === 
        
        encoder_states, encoder_h_n, memory, mu, logvar, embedded_cognateset, source_padding_mask = self.encode_and_prepare_decode(source_tokens, source_langs, source_seqs_lens, inject_embedding)
        
        encoder_states = self.dropout(encoder_states)
              
        
        # === INITIAL DECODE ===
                
        decoder_output, hidden, attention_weights, (_bos_tkn_vec, _sep_lang_vec) = self.initial_decode(N, memory, encoder_h_n, encoder_states, embedded_cognateset, source_padding_mask)
        
        decoder_output = self.dropout(decoder_output)


        # === DECODE SEQUENCE ===
        
        scores = []
        decoder_states_list = []
        
        # no need to decode first token since we already fed in <bos>
        for target_token in target_tokens[:, 1:].T: # for (N) in (L_target - 1, N)
            
            # == CLASSIFY ==
                                    
            classification_input = decoder_output + attention_weights
            # classification_input: (N, L = 1, E)
            
            match (self.prompt_mlp_with_one_hot_lang, self.gated_mlp_by_target_lang):
                case (True, True):
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                    tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
                case (True, False):
                    classification_input = torch.cat((classification_input, self.get_one_hot_target_lang(target_langs)), dim=-1)
                    tkn_scores = self.mlp(classification_input)
                case (False, True):
                    tkn_scores = self.mlp(rearrange(classification_input, 'N 1 E -> N E'), target_langs - self.min_possible_target_lang_langs_idx)
                    tkn_scores = rearrange(tkn_scores, 'N V -> N 1 V')
                case (False, False):
                    tkn_scores = self.mlp(classification_input)
            # tkn_scores: (N, 1, V) this is before softmax
            
            scores.append(tkn_scores)
            
            # == ONTO NEXT RECURRENCE ==
            
            decoder_states_list.append(decoder_output)
            
            target_tkn_vec = rearrange(target_token, 'N -> N 1')
            target_lang_vec = target_langs

            decoder_output, hidden, attention_weights = self.decode_once(target_tkn_vec, target_lang_vec if self.lang_embedding_when_decoder else None, attention_weights, hidden, encoder_states, embedded_cognateset, source_padding_mask)
            
            decoder_output = self.dropout(decoder_output)

        # by now, scores is list (same len as target tokens) of tensors (shape (N, 1, ipa_vocab len))
        logits = torch.stack(scores).squeeze(2).swapaxes(0,1)
        # logits: (N, T, ipa_vocab len)
        decoder_states = torch.stack(decoder_states_list).squeeze(2).swapaxes(0,1)
        # decoder_states (N, T, H)
        
        result_dict = {
            "logits": logits,
            "encoder_states": encoder_states, # if needed for whatever reason
            "decoder_states": decoder_states,
            "encoder_h_n": encoder_h_n,
        }
        
        if self.use_vae_latent:
            result_dict['mu'] = mu
            result_dict['logvar'] = logvar 

        return result_dict

    # get the probability of a speculated output sequence given some source sequence to encode
    # essentially the same as teacher forcing on the speculated sequence and then taking the product of the probabilities
    def get_sequence_log_probs(self,
        source_tokens: Tensor, # (N, L)
        source_langs: Tensor | None, # (N, L)
        source_seqs_lens: Tensor, # (N, 1)
        speculated_target_tokens: Tensor,
        target_langs: Tensor | None,
    ):
        N = source_tokens.size(0) # viz. batch size
        teacher_forcing_decode_res = self.teacher_forcing_decode(source_tokens, source_langs, source_seqs_lens, speculated_target_tokens, target_langs)
        
        speculated_target_tokens_mask: Tensor = (speculated_target_tokens != PAD_IDX)
        
        logits = teacher_forcing_decode_res['logits']
        log_probs = torch.log_softmax(logits, dim=-1)
        # Logits, log_probs: (N, L_target, V)
                        
        log_probs_seq_ = torch.diagonal(log_probs[:, torch.arange(log_probs.shape[1]), speculated_target_tokens[:, 1:]]).T
        log_probs_seq = log_probs_seq_ * speculated_target_tokens_mask[:, 1:]
        
        log_prob_sum = torch.sum(log_probs_seq, dim=-1)
        
        return log_probs_seq, log_prob_sum, logits




    def unpack_batch(self, batch) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        match self.training_mode:
            case 'encodeWithLangEmbedding':
                (d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), _, _ = batch   
                
                source_tokens = d_cat_tkns
                source_langs = d_cat_langs
                source_seqs_lens = d_cat_lens
                target_tokens = p_tkns

                N = source_tokens.shape[0]
                target_lang_ipa_ids = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
                target_lang_lang_ids = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
                
            case 'encodeTokenSeqAppendTargetLangToken':
                _ , (source_tokens, source_seqs_lens, target_lang_ipa_ids, target_lang_lang_ids, target_tokens), _ = batch
                
                source_langs = None
            case _:
                raise NotImplemented

        # source_tokens: (N, L)
        # source_langs: (N, L) | None
        # target_tokens: (N, L_target)
        # target_lang_ipa_ids: (N, 1) | None
        # target_lang_lang_ids: (N, 1) | None
        return source_tokens, source_langs, source_seqs_lens, target_tokens, target_lang_ipa_ids, target_lang_lang_ids

    def reunpack_batch_for_reranking(self, batch):
        assert self.training_mode == 'encodeWithLangEmbedding' # make sure it's d2p (?)
        # WARNING: this assumes the batch[0] is from d2p and batch[1] is from p2d. it also assumes N = 1 on d2p end.
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, _d_indv_lens, _p_lang_lang_vec, proto_seq, p_l_tkns, p_fs), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs), _ = batch
        
        N = daughters_concat_seq.shape[0]
        assert N == 1
        proto_lang_ipa_ids = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        proto_lang_lang_ids = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)

        return (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq, proto_lang_ipa_ids, proto_lang_lang_ids), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs)

    def rereunpack_batch_for_batched_reranking(self, batch):
        assert self.training_mode == 'encodeWithLangEmbedding' # make sure it's d2p (?)
        # WARNING: this assumes the batch[0] is from d2p and batch[1] is from p2d. it also assumes N = 1 on d2p end.
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, _d_indv_lens, _p_lang_lang_vec, proto_seq, p_l_tkns, p_fs), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs), (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = batch
        
        # prompted_proto_seqs_s (N, Nd, L_p)
        # prompted_proto_seqs_lens_s (N, Nd, 1)
        # daughters_ipa_langs_s (N, Nd, 1)
        # daughters_lang_langs_s (N, Nd, 1)
        # daughters_seqs_s (N, Nd, L_d)

        N = daughters_concat_seq.shape[0]
        proto_lang_ipa_ids = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        proto_lang_lang_ids = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)

        return (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq, proto_lang_ipa_ids, proto_lang_lang_ids), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs), (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s)



    def training_step(self, batch, _batch_idx):
        
        source_tokens, source_langs, source_seqs_lens, target_tokens, _target_langs_ipa, target_langs_lang = self.unpack_batch(batch)
        
        decode_res = self.teacher_forcing_decode(source_tokens, source_langs, source_seqs_lens, target_tokens, target_langs_lang)
        
        loss, recon_loss, kl_loss = self.get_loss_from_teacher_forcing_decode_res(decode_res, target_tokens)
                
        if self.enable_logging:
            self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
            self.log(f"{self.logger_prefix}/train/recon_loss", recon_loss)
            if kl_loss != None:
                self.log(f"{self.logger_prefix}/train/kl_loss", kl_loss)
            self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
                         
        return loss
    
    def get_loss_from_teacher_forcing_decode_res(self, decode_res, target_tokens):
        if self.use_vae_latent:
            logits, mu, logvar = decode_res['logits'], decode_res['mu'], decode_res['logvar']
            
            recon_loss, kl_loss = utils.calc_vae_losses(
                logits.swapaxes(1,2), 
                target_tokens[:, 1:], # (N, T)
                mu,
                logvar
            )
            loss = recon_loss + kl_loss
        else:
            logits: Tensor = decode_res['logits']
            # logits: (N, T, ipa_vocab len)
            
            recon_loss = utils.calc_cross_entropy_loss(
                logits.swapaxes(1,2), # (N, ipa_vocab len, T)
                target_tokens[:, 1:] # (N, T)
            ) # sth like tensor(5.4489, grad_fn=<NllLoss2DBackward0>)
            kl_loss = torch.tensor(0.0)
            loss = recon_loss
        
        assert(loss != None)
        return loss, recon_loss, kl_loss

    def test_step(self, batch, batch_idx):
        if self.transductive_test:
            return self.shared_eval_step(batch, batch_idx, prefix='transductive')
        else:
            return self.shared_eval_step(batch, batch_idx, prefix='test')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='val')
    
    def forward_on_batch(self, batch):
        # 1 > unpacking
        source_tokens, source_langs, source_seqs_lens, target_tokens, _target_langs_ipa, target_langs_lang = self.unpack_batch(batch)

        # 2 > predict
        decode_res = self.teacher_forcing_decode(source_tokens, source_langs, source_seqs_lens, target_tokens, target_langs_lang)

        # 3 > loss
        loss, recon_loss, kl_loss = self.get_loss_from_teacher_forcing_decode_res(decode_res, target_tokens)

        return decode_res['logits'], loss, recon_loss, kl_loss


    def forward(self, source_tokens, source_langs, source_seqs_lens, target_langs, batch) -> Tensor:
        
        processed_endnodes = None
        match self.decode_mode:
            case 'greedy_search':
                predictions = self.greedy_decode(source_tokens, source_langs, source_seqs_lens, target_langs)
            case 'beam_search':
                _beam_sequences, _beam_log_probs, _lenient_endnodes, processed_endnodes, predictions = self.beam_search_decode(source_tokens, source_langs, source_seqs_lens, target_langs, beam_size=self.beam_size)
            case 'reranked_beam_search':
                assert 'reranker' in self.__dict__ and 'rescorer' in self.__dict__, 'reranker and rescorer must be set before calling reranked_beam_search_decode'
                (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq, proto_lang_ipa_ids, proto_lang_lang_ids), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs) = self.reunpack_batch_for_reranking(batch)

                _rescored_endnodes, predictions = self.reranked_beam_search_decode(
                    daughters_concat_seq, 
                    daughters_concat_langs, 
                    daughters_concat_seqs_lens, 
                    proto_lang_lang_ids,
                    self.beam_size, 
                    daughter_seqs, 
                    target_ipa_langs, 
                    target_lang_langs, 
                    self.reranker, 
                    self.rescorer
                )
            case 'batched_reranked_beam_search':
                assert 'reranker' in self.__dict__ and 'rescorer' in self.__dict__, 'reranker and rescorer must be set before calling reranked_beam_search_decode'
                (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq, proto_lang_ipa_ids, proto_lang_lang_ids), _, (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = self.rereunpack_batch_for_batched_reranking(batch)
                
                # prompted_proto_seqs_s (N, Nd, L_p)
                # prompted_proto_seqs_lens_s (N, Nd, 1)
                # daughters_ipa_langs_s (N, Nd, 1)
                # daughters_lang_langs_s (N, Nd, 1)
                # daughters_seqs_s (N, Nd, L_d)

                _rescored_endnodes, predictions = self.batched_reranked_beam_search_decode(
                    daughters_concat_seq, 
                    daughters_concat_langs, 
                    daughters_concat_seqs_lens, 
                    proto_lang_lang_ids,
                    self.beam_size, 
                    daughters_seqs_s, # (N, Nd, L_d)
                    daughters_ipa_langs_s, # (N, Nd, 1)
                    daughters_lang_langs_s, # (N, Nd, 1)
                    self.reranker, 
                    self.rescorer
                )
            case _: raise NotImplemented
        
        return predictions, processed_endnodes


    def shared_eval_step(self, batch, _batch_idx, prefix: str):
        source_tokens, source_langs, source_seqs_lens, target_tokens, _target_langs_ipa, target_langs_lang = self.unpack_batch(batch)
        N = source_tokens.size(0)
        target_langs = target_langs_lang
        self.batch_size_check(N, source_tokens, source_langs, target_tokens, target_langs)
        
        # === get loss ===
        
        decode_res = self.teacher_forcing_decode(source_tokens, source_langs, source_seqs_lens, target_tokens, target_langs)
        loss, recon_loss, kl_loss = self.get_loss_from_teacher_forcing_decode_res(decode_res, target_tokens)
        
        losses_res_dict = {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
        for k in losses_res_dict:
            self.log(f"{self.logger_prefix}/{prefix}/{k}", losses_res_dict[k], on_step=False, on_epoch=True, batch_size=N)
        
        # === get predictions ===
        predictions, processed_endnodes = self.forward(source_tokens, source_langs, source_seqs_lens, target_langs, batch)
        
        # === make strings ===
        string_res = utils.mk_strings_from_forward(self, source_tokens, source_langs, target_tokens, target_langs, predictions, processed_endnodes)
        
        for string_res_dict in string_res:        
            self.evaled_on_target_langs.add(string_res_dict['target_lang'])
            self.eval_step_outputs.append(string_res_dict)


    def on_validation_epoch_end(self):
        return self.shared_eval_epoch_end('val')
    
    def on_test_epoch_end(self):
        if self.transductive_test:
            return self.shared_eval_epoch_end('transductive')
        else:
            return self.shared_eval_epoch_end('test')


    def shared_eval_epoch_end(self, prefix: str):
        self.evaled_on_target_langs.add(ALL_TARGET_LANGS_LABEL)
        
        metric_out = utils.calc_metrics_from_string_dicts(
            self,
            self.evaled_on_target_langs, 
            self.eval_step_outputs,
            self.all_lang_summary_only if prefix == 'val' else False,
            prefix,
        )
        
        # reset stuff
        self.evaled_on_target_langs.clear()
        self.eval_step_outputs.clear()
        
        return metric_out