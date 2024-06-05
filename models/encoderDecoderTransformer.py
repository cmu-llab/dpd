from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.rescoring import Reranker, Rescorer
    from lib.rescoring import BatchedGreedyBasedRerankerBase, BatchedReranker, BatchedRescorer

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchshow as ts
from specialtokens import *
from prelude import *
from .partials.embedding import Embedding
from .partials.mlp import MLP
from .partials.attention import Attention
from lib.vocab import Vocab
from einops import rearrange, repeat
import models.utils as utils
import transformers

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embedding_dim, 2)* math.log(10000) / embedding_dim)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, embedding_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self._dropout = nn.Dropout(dropout)
        self.register_buffer('_pos_embedding', pos_embedding)
        self._pos_embedding: Tensor

    def forward(self, token_embedding, lens=None):
        # if True:
        if lens is None:  # each sequence is one sequence (target)
            return self._dropout(token_embedding + self._pos_embedding[:, :token_embedding.size(1), :token_embedding.size(2)])
        else:  # each sequence is a concatenation of multiple sequences (source) TODO what's going on here?
            pos_embedding = []
            for _lens in lens:
                _pos_embedding = []
                for len in _lens:
                    _pos_embedding.append(self._pos_embedding[:, :len, :token_embedding.size(2)])  # positional encoding is applied per subsequence
                _pos_embedding = torch.cat(_pos_embedding, dim=1)  # subsequence-wise positional encodings concatenated
                pos_embedding.append(torch.squeeze(_pos_embedding, dim=0))
            pos_embedding = pad_sequence(pos_embedding, batch_first=True, padding_value=0)
            return self._dropout(token_embedding + pos_embedding)

class EncoderDecoderTransformer(pl.LightningModule):
    def __init__(self,
        ipa_vocab: Vocab,
        lang_vocab: Vocab,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embedding_dim: int,
        nhead: int,
        feedforward_dim: int,
        dropout_p: float,
        max_len: int,
        logger_prefix: str,
        task: str, # 'd2p' | 'p2d'
        inference_decode_max_length: int,
        all_lang_summary_only: bool,

        use_xavier_init: bool,
        lr: float,
        warmup_epochs: int,
        max_epochs: int,
        weight_decay: float,
        
        beta1: float,
        beta2: float,
        eps: float,
        
        init_ipa_embedding: None | TokenEmbedding,
        init_lang_embedding: None | TokenEmbedding,
    ):
        super().__init__()
        self.ipa_vocab = ipa_vocab
        self.lang_vocab = lang_vocab
        self.protolang: str = lang_vocab.protolang # like 'Middle Chinese (Baxter and Sagart 2014)'
        self.use_xavier_init = use_xavier_init
        self.logger_prefix = logger_prefix
        self.task = task
        self.inference_decode_max_length = inference_decode_max_length
        self.all_lang_summary_only = all_lang_summary_only
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.embedding_dim = embedding_dim

        model_dim = embedding_dim  # transformer d_model
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        
        self.transformer: nn.Transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.generator = nn.Linear(model_dim, len(ipa_vocab))

        self.ipa_embedding: TokenEmbedding
        self.lang_embedding: TokenEmbedding
        match init_ipa_embedding:
            case None:
                self.ipa_embedding = TokenEmbedding(len(ipa_vocab), embedding_dim)
            case _:
                self.ipa_embedding = init_ipa_embedding
        match init_lang_embedding:
            case None:
                self.lang_embedding = TokenEmbedding(len(lang_vocab), model_dim)
            case _:
                self.lang_embedding = init_lang_embedding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout_p, max_len)
        
        if self.use_xavier_init:
            print("performing Xavier initialization")
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = lr,
            betas = (beta1, beta2), # default
            eps = eps, # default
            weight_decay=weight_decay,
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
        
        self.transductive_test = False
        
        self.ipa_embedding.embedding.weight.data[PAD_IDX].fill_(0)
        self.lang_embedding.embedding.weight.data[PAD_IDX].fill_(0)
        
        self.eval_step_outputs = []
        self.evaled_on_target_langs = set() # keep track of which target langs we've evaled on if we're using encodeTokenSeqAppendTargetLangToken
        
        # reranking support
        self.possible_target_lang_langs = self.lang_vocab.to_indices(self.lang_vocab.daughter_langs) # list of ints, indices in lang_vocab.
        self.min_possible_target_lang_langs_idx = min(self.possible_target_lang_langs)


    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler_config
        }
            
    def forward(
        self,
        s_tkns,
        s_langs,
        s_lens,
        s_mask,
        s_pad_mask,
        memory_key_pad_mask, # (N, L) what positions to ignore in memory when doing cross attention?, usually same as s_pad_mask
        t_tkns,
        t_mask,
        t_pad_mask,
        inject_embedding: Tensor | None = None, # replace embedding fed into encoder
    ):
        memory = self.encode(s_tkns, s_lens, s_langs, s_mask, s_pad_mask, inject_embedding=inject_embedding)
        decoder_out = self.decode2(t_tkns, memory, t_mask, t_pad_mask, memory_key_pad_mask)
        logits = self.generator(decoder_out)
        return logits, decoder_out # (N, L, V)

    # get the probability of a speculated output sequence given some source sequence to encode
    # essentially the same as forward on the speculated sequence and then taking the product of the probabilities
    def get_sequence_log_probs(self,
        batch,
        speculated_target_tokens: Tensor,
    ):
        assert self.task == 'd2p'
        (d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns,  p_l_tkns, p_fs), foo1, foo2 = batch
        
        N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(((d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, speculated_target_tokens,  p_l_tkns, p_fs), foo1, foo2), self.device, self.task, self.ipa_vocab, self.lang_vocab, self.protolang)
        
        # 2 > preprocess
        
        # can't have dummy target mask with every token being pad
        t_not_dummy_mask = ~(t_pad_mask.all(dim=1).unsqueeze(dim=1)) # (N, 1) where True indicates non-dummy
        # t_pad_mask shall only apply if a True is present in t_not_dummy_mask
        s_not_dummy_mask = ~(s_pad_mask.all(dim=1).unsqueeze(dim=1)) # (N, 1) similar

        # 3 > predict
        s_tkns, s_langs, s_indv_lens, s_pad_mask, s_pad_mask, s_not_dummy_mask = s_tkns.to(self.device), s_langs.to(self.device), s_indv_lens.to(self.device), s_pad_mask.to(self.device), s_pad_mask.to(self.device), s_not_dummy_mask.to(self.device)

        logits, decoder_out = self.forward(
            s_tkns = s_tkns,
            s_langs = s_langs,
            s_lens = s_indv_lens,
            s_mask = s_mask,
            s_pad_mask = s_pad_mask * s_not_dummy_mask,
            memory_key_pad_mask = s_pad_mask * s_not_dummy_mask,
            t_tkns = t_tkns_in,
            t_mask = t_mask,
            t_pad_mask = t_pad_mask * t_not_dummy_mask,
        )
                
        speculated_target_tokens_mask: Tensor = (speculated_target_tokens != PAD_IDX)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        # logits, log_probs: (N, L_target, V)
                        
        log_probs_seq_ = torch.diagonal(log_probs[:, torch.arange(log_probs.shape[1]), speculated_target_tokens[:, 1:]]).T
        log_probs_seq = log_probs_seq_ * speculated_target_tokens_mask[:, 1:]
        
        log_prob_sum = torch.sum(log_probs_seq, dim=-1)
        
        return log_probs_seq, log_prob_sum, logits


    def forward_on_batch(self, 
        batch,
        inject_embedding: Tensor | None = None, # replace embedding fed into encoder
    ):
        
        # 1 > unpacking
        
        N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(batch, self.device, self.task, self.ipa_vocab, self.lang_vocab, self.protolang)
        
        # 2 > preprocess
        
        # can't have dummy target mask with every token being pad
        t_not_dummy_mask = ~(t_pad_mask.all(dim=1).unsqueeze(dim=1)) # (N, 1) where True indicates non-dummy
        # t_pad_mask shall only apply if a True is present in t_not_dummy_mask
        s_not_dummy_mask = ~(s_pad_mask.all(dim=1).unsqueeze(dim=1)) # (N, 1) similar

        # 3 > predict

        logits, decoder_out = self.forward(
            s_tkns = s_tkns,
            s_langs = s_langs,
            s_lens = s_indv_lens,
            s_mask = s_mask,
            s_pad_mask = s_pad_mask * s_not_dummy_mask,
            memory_key_pad_mask = s_pad_mask * s_not_dummy_mask,
            t_tkns = t_tkns_in,
            t_mask = t_mask,
            t_pad_mask = t_pad_mask * t_not_dummy_mask,
            inject_embedding = inject_embedding,
        )
        
        # 4 > loss

        loss = self.loss_fn(
            rearrange(logits, 'N L V -> (N L) V'),
            rearrange(t_tkns_out, 'N L -> (N L)')
        )
        
        return logits, loss, decoder_out

    def greedy_decode(self, s_tkns, s_lens, s_langs, s_mask, s_pad_mask, decode_max_len):
        memory = self.encode(s_tkns, s_lens, s_langs, s_mask, s_pad_mask)
        # memory: (N L size)
        
        generated_sequences = []
        N = s_tkns.shape[0]
        reached_eos = torch.zeros((N, 1), dtype=torch.bool).to(self.device)
        ys = torch.zeros(N, 1).fill_(BOS_IDX).type_as(s_tkns.data) # (N, L)

        for i in range(decode_max_len - 1):

            out = self.decode1(
                ys, 
                memory, 
                nn.Transformer.generate_square_subsequent_mask(ys.size(1)).bool().to(self.device)
            )
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            pred_tkn = (reached_eos == 1) * PAD_IDX | (reached_eos == 0) * rearrange(next_word, "N -> N 1")
            reached_eos = reached_eos | (pred_tkn == EOS_IDX)

            ys = torch.cat([ys, pred_tkn], dim=1)

            if torch.all(reached_eos == 1):
                break

        predictions = ys # (N, Ldecoded)
        
        return predictions
   

        # region  original, unbatched decode
        for source_memory in memory:
            latent_ys = torch.ones(1, 1).fill_(BOS_IDX).long().to(self.device)
            for i in range(decode_max_len - 1):
                target_mask = nn.Transformer.generate_square_subsequent_mask(latent_ys.size(1)).bool().to(self.device)
                out = self.decode1(latent_ys, torch.unsqueeze(source_memory, dim=0), target_mask)
                prob = self.generator(out[:, -1])
                _, latent_next_idx = torch.max(prob, dim=1)
                latent_next_idx = latent_next_idx.item()

                latent_ys = torch.cat((latent_ys, torch.ones(1, 1).fill_(latent_next_idx).long().to(self.device)), dim=1)
                if latent_next_idx == EOS_IDX:
                    break
            generated_sequence = torch.squeeze(latent_ys)
            generated_sequences.append(generated_sequence)
        
        predictions = pad_sequence(generated_sequences, batch_first=True, padding_value=PAD_IDX)
        # endregion

        return predictions # (N, L)

    def training_step(self, batch, batch_idx):
        
        logits, loss, _decoder_out = self.forward_on_batch(batch)
        
        self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
        self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)

        return loss

    def embed(self, s_tkns, s_langs, s_lens):# -> Any:
        s_tkn_emb = self.pos_encoding(self.ipa_embedding(s_tkns), s_lens)
        s_lang_emb = self.lang_embedding(s_langs) if s_langs is not None else torch.zeros_like(s_tkn_emb)
        s_emb = s_tkn_emb + s_lang_emb
        return s_emb

    def encode(self, 
        s_tkns: Tensor, # (N, L)
        s_lens: Tensor | None, # (N, 1)
        s_langs: Tensor | None, # (N, 1)
        s_mask: Tensor | None, # usually None since not masking encode
        s_pad_mask: Tensor, # (N, L)
        inject_embedding: Tensor | None = None, # replace embedding fed into encoder
    ):
        # s_tkn_emb = self.pos_encoding(self.ipa_embedding(s_tkns), s_lens)
        # s_lang_emb = self.lang_embedding(s_langs) if s_langs is not None else torch.zeros_like(s_tkn_emb)
        # s_emb = s_tkn_emb + s_lang_emb
        if inject_embedding == None:        
            s_emb = self.embed(s_tkns, s_langs, s_lens)
        else:
            s_emb = inject_embedding
        
        memory = self.transformer.encoder(
            s_emb,
            mask=s_mask,
            src_key_padding_mask=s_pad_mask
        )
        return memory # (N, L, E)

    # why no key mask here?
    def decode1(self, 
        t_tkns, # (N, L)
        memory, 
        t_mask, # (N, L)
    ):
        t_tkn_emb = self.pos_encoding(self.ipa_embedding(t_tkns))
        t_emb = t_tkn_emb

        decoder_out = self.transformer.decoder(
            t_emb, # (N, L, E)
            memory, #
            tgt_mask=t_mask # (N, L)
        )
        
        return decoder_out
    
    def decode2(self, 
        t_tkns, # (N, L)
        memory, 
        t_mask, # (N, L)
        t_pad_mask,
        memory_key_pad_mask,
    ):
        t_tkn_emb = self.pos_encoding(self.ipa_embedding(t_tkns))
        t_emb = t_tkn_emb

        decoder_out = self.transformer.decoder(
            t_emb, # (N, L, E)
            memory, #
            tgt_mask=t_mask, # (N, L)
            tgt_key_padding_mask=t_pad_mask,
            memory_key_padding_mask=memory_key_pad_mask,
        )
        
        return decoder_out # (N, L, E)
    
    def test_step(self, batch, batch_idx):
        if self.transductive_test:
            return self.shared_eval_step(batch, batch_idx, prefix='transductive')
        else:
            return self.shared_eval_step(batch, batch_idx, prefix='test')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, batch_idx, prefix='val')

    def shared_eval_step(self, batch, _batch_idx, prefix: str):
        N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(batch, self.device, self.task, self.ipa_vocab, self.lang_vocab, self.protolang)

        # 1 > get loss
        
        _logits, loss, _decoder_out = self.forward_on_batch(batch)
        self.log(f"{self.logger_prefix}/{prefix}/loss", loss, on_step=False, on_epoch=True, batch_size=N)
        
        # 2 > make predictions
        
        predictions = self.greedy_decode(s_tkns, s_indv_lens, s_langs, s_mask, s_pad_mask, decode_max_len=self.inference_decode_max_length)
        
        # 3 > make strings
        
        string_res = utils.mk_strings_from_forward(self, s_tkns, s_langs, t_tkns, target_langs = t_lang_lang, predictions=predictions, processed_endnodes=None)
        
        for string_res_dict in string_res:        
            self.evaled_on_target_langs.add(string_res_dict['target_lang'])
            self.eval_step_outputs.append(string_res_dict)

        return None
    
    
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
            self.all_lang_summary_only,
            prefix,
        )
        
        # reset stuff
        self.evaled_on_target_langs.clear()
        self.eval_step_outputs.clear()
        
        return metric_out
