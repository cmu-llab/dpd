from __future__ import annotations
from typing import Annotated
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.biDirReconIntegration import biDirReconModelRNN, biDirReconModelTrans
    
from models.encoderDecoderRNN import Seq2SeqRNN # used as d2p
import torch
import pytorch_lightning as pl
from torch import Tensor

from specialtokens import *

from models.partials.embedding import Embedding
from models.partials.mlp import MLP
from models.partials.attention import Attention
from models.partials.rnn import GRURNN
from models.partials.embeddingPredictionNet import EmbeddingPredictionNet
from lib.vocab import Vocab
from einops import repeat, rearrange
from lib.tensor_utils import sequences_equal, num_sequences_equal
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch.nn as nn
from models.utils import CringeLoss
import torch.nn.functional as F
from einops import rearrange, repeat
import models.utils as utils

from prelude import *

# used to embed a strategy inside biDirReconModel, disregarding _fake_self and treating biDirReconModel as self. think of the methods inside a biDirReconModel, so a different self is used to access strategy-specific parameters
class BidirTrainStrategyBase():
    def __init__(self) -> None:
        pass
    
    # add stuff to biDirReconModel, etc. Always called before training
    def extra_init(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans):
        pass
    
    def unpack_batch(self, batch) -> batch_t:
        d2p_batch, p2d_batch, structured_p2d_batch = batch
        
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq) = d2p_batch
        (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs) = p2d_batch
        (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = structured_p2d_batch
        
        return (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs), (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s)

    def training_step(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans, batch: batch_t, batch_idx: int) -> Tensor:
        raise NotImplemented
    
    def on_train_epoch_end(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans) -> dict | None:
        raise NotImplemented
    
    def validation_step(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans, batch: batch_t, batch_idx: int) -> Tensor:
        raise NotImplemented
    
    def on_validation_epoch_end(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans) -> dict | None:
        raise NotImplemented

    def test_step(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans, batch: batch_t, batch_idx: int) -> Tensor:
        raise NotImplemented
    
    def on_testepoch_end(_fake_self, self: biDirReconModelRNN | biDirReconModelTrans) -> dict | None:
        raise NotImplemented

# === beam strats ===

class BeamSampleStrategyBase(BidirTrainStrategyBase):
    def __init__(self, 
        beam_negative_sample_k: int,
    ) -> None:
        super().__init__()
        self.beam_negative_sample_k = beam_negative_sample_k

    def get_negative_samples_from_d2p(strat, self: biDirReconModelRNN | biDirReconModelTrans, batch, 
        top_k_incorrect: int, # want 5 bad and 1 good
        top_k_buffer: int, # beam search few more so we don't run out
    ):
        
        incorrect_proto_sets = []
        correct_protos = []
        self.d2p.eval()
        
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq), (prompted_proto_seq, prompted_proto_seqs_lens, target_ipa_langs, target_lang_langs, daughter_seqs), (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = batch
        
        N = daughters_concat_seq.shape[0]
        proto_lang_lang_ids = repeat((torch.LongTensor([self.d2p.lang_vocab.get_idx(self.d2p.protolang)]).to(self.d2p.device)), '1 -> N 1', N=N)
        
        beam_sequences, beam_log_probs, lenient_endnodes, processed_endnodes, best_seqs_padded = self.d2p.beam_search_decode(daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_lang_lang_ids, beam_size=3)
        
        for beam_sequences_for_batch, ranked_endnodes, gold_proto in zip(beam_sequences, processed_endnodes, proto_seq):
            predicted_incorrect_protos = []
            
            for entry in ranked_endnodes:
                predicted_proto = entry['endnode_seq']
                correct = sequences_equal(rearrange(predicted_proto, 'L -> 1 L'), rearrange(gold_proto, 'L -> 1 L'))
                if not correct:
                    predicted_incorrect_protos.append(predicted_proto)
            
            while len(predicted_incorrect_protos) < top_k_incorrect:
                for seq in beam_sequences_for_batch:
                    for entry in ranked_endnodes:
                        predicted_proto = entry['endnode_seq']
                        if not sequences_equal(predicted_proto.unsqueeze(0), seq.unsqueeze(0)):
                            predicted_incorrect_protos.append(predicted_proto)
                    
            predicted_incorrect_protos = predicted_incorrect_protos[:top_k_incorrect]
            assert len(predicted_incorrect_protos) == top_k_incorrect
            incorrect_proto_sets.append(predicted_incorrect_protos)
            
        correct_protos.extend(list(proto_seq))
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        incorrect_proto_sets_s = rearrange(pad_sequence(flatten(incorrect_proto_sets), batch_first=True, padding_value=PAD_IDX), '(N topK) L -> N topK L', N = N)
        
        # incorrect_proto_sets: list[N] of list[top_k_incorrect] of tensors (1, L_p_hat)
        # incorrect_proto_sets_s: (N, top_k_incorrect, L_p_hat_max)
        # correct_protos: list[N] of tensors (1, L_p)
        return incorrect_proto_sets, incorrect_proto_sets_s, correct_protos

    def recon_proto_and_recon_daughter(strat, self: biDirReconModelRNN | biDirReconModelTrans, batch):
        
        # assumes batch[0] is d2p and batch[1] is from p2d AND they correspond to same cognate set (bug alert)
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq), _, (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = batch

        # prompted_proto_seqs_s (N, Nd, L_p)
        # prompted_proto_seqs_lens_s (N, Nd, 1)
        # daughters_ipa_langs_s (N, Nd, 1)
        # daughters_lang_langs_s (N, Nd, 1)
        # daughters_seqs_s (N, Nd, L_d)
        
        N = daughters_concat_seq.shape[0]
        Nd = daughters_seqs_s.shape[1]
        
        proto_lang_ipa_ids = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        proto_lang_lang_ids = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        
        
        # === daughter to proto beam search ===
        # (no training! only to get a predicted proto)
        
        k = strat.beam_negative_sample_k
        top_k_buffer = k
        
        incorrect_proto_sets, incorrect_proto_sets_s, correct_protos = strat.get_negative_samples_from_d2p(self, batch, top_k_incorrect = k, top_k_buffer = top_k_buffer)
        # incorrect_proto_sets: list[N] of list[k] of tensors (1, L_p_hat)
        # incorrect_proto_sets_s: (N, k, L_p_hat_max)
        # correct_protos: list[N] of tensors (1, L_p)
        
        target_prompted_bad_protos = torch.cat((
            repeat(daughters_ipa_langs_s, 'N Nd 1 -> N k Nd 1', k=k), 
            repeat(incorrect_proto_sets_s, 'N k L -> N k Nd L', Nd=Nd)
        ), dim=-1)
        # target_prompted_bad_protos (N, k, Nd, L_bad_max+1)

        target_prompted_good_protos = repeat(prompted_proto_seqs_s, 'N Nd L -> N k Nd L', k=k)
        # target_prompted_good_protos (N, k, Nd, L_good_max+1)
        
        Lb = target_prompted_bad_protos.shape[-1]
        Lg = target_prompted_good_protos.shape[-1]
        good_bad_target_prompted_protos_ensemble = torch.cat((
            torch.cat((
                rearrange(target_prompted_bad_protos, 'N k Nd Lb -> 1 N k Nd Lb'),
                torch.ones(1, N, k, Nd, max(Lg - Lb, 0), dtype=torch.long).to(self.device) * PAD_IDX
            ), dim=-1), 
            torch.cat((
                rearrange(target_prompted_good_protos, 'N k Nd Lg -> 1 N k Nd Lg'),
                torch.ones(1, N, k, Nd, max(Lb - Lg, 0), dtype=torch.long).to(self.device) * PAD_IDX
            ), dim=-1)
        ), dim=0) # (2, N, k, Nd, L)
        
        proto_correct = torch.cat((
            torch.zeros((1, N, k, Nd, 1), dtype=torch.long).to(self.device),
            torch.ones((1, N, k, Nd, 1), dtype=torch.long).to(self.device)
        ), dim=0) # (2, N, k, Nd, 1)
        
        
        # === proto to daughter recon ===
        
        valid_daughters_mask = (daughters_ipa_langs_s != PAD_IDX) # (N, Nd, 1)
        broadcasted_valid_daughters_mask = repeat(repeat((daughters_ipa_langs_s != PAD_IDX), 'N Nd 1 -> 2 N Nd 1'), 'c2 N Nd 1 -> c2 N k Nd 1', k = k)
        # broadcasted_valid_daughters_mask: (2, N, k, d, 1)
        
        broadcasted_daughters_lang_langs_s = None if daughters_lang_langs_s == None else repeat(repeat(daughters_lang_langs_s, 'N Nd 1 -> 2 N Nd 1'), 'c2 N Nd 1 -> c2 N k Nd 1', k = k) # (2, N, k, d, 1)
        
        broadcasted_daughters_seqs_s = repeat(repeat(daughters_seqs_s, 'N Nd L -> 2 N Nd L'), 'c2 N Nd L -> c2 N k Nd L', k = k) # (2, N, k, d, L_target)

        good_bad_target_prompted_protos_ensemble_lens = torch.sum(good_bad_target_prompted_protos_ensemble != PAD_IDX, dim=-1).unsqueeze(-1) + (1 * (~ broadcasted_valid_daughters_mask)) # dummy length 1 for sequence of pad
        # good_bad_target_prompted_protos_ensemble_lens: (2, N, k, d, 1)
                
        broadcasted_daughters_lang_langs_s_with_dummy_for_pad = None if broadcasted_daughters_lang_langs_s == None else broadcasted_daughters_lang_langs_s + ((1 * (~ broadcasted_valid_daughters_mask)) * self.p2d.min_possible_target_lang_langs_idx) # (2, N, k, d, 1)
        
        p2d_decode_res = self.p2d.teacher_forcing_decode(
            source_tokens = rearrange(
                good_bad_target_prompted_protos_ensemble,
                'c2 N k Nd L -> (c2 N k Nd) L'
            ),
            source_langs = None, 
            source_seqs_lens = rearrange(
                good_bad_target_prompted_protos_ensemble_lens, 
                'c2 N k Nd 1 -> (c2 N k Nd) 1'
            ),
            target_tokens = rearrange(
                broadcasted_daughters_seqs_s, 
                'c2 N k Nd L -> (c2 N k Nd) L'
            ),
            target_langs = rearrange(
                broadcasted_daughters_lang_langs_s_with_dummy_for_pad, 
                'c2 N k Nd 1 -> (c2 N k Nd) 1'
            )
        )
                
        p2d_decode_logits = p2d_decode_res['logits'] # ((c2 N k Nd), L_d-1, V)
        p2d_decode_logits_s = rearrange(
            p2d_decode_logits, 
            '(c2 N k Nd) L V -> c2 N k Nd L V', c2=2, N=N, k=k, Nd=Nd
        ) # (2, N, k, Nd, L, V)
        
        p2d_decode_logits_s_masked = p2d_decode_logits_s * rearrange(broadcasted_valid_daughters_mask, 'c2 N k Nd 1 -> c2 N k Nd 1 1') # (2, N, k, Nd, L, V)
        
        return N, Nd, k, proto_correct, good_bad_target_prompted_protos_ensemble, p2d_decode_res, broadcasted_valid_daughters_mask, broadcasted_daughters_seqs_s, p2d_decode_logits_s_masked
        # proto_correct (2, N, k, Nd, 1)
        # good_bad_target_prompted_protos_ensemble (2, N, k, Nd, L)
        # broadcasted_valid_daughters_mask (2, N, k, d, 1)
        # broadcasted_daughters_seqs_s (2, N, k, d, L_target)
        # p2d_decode_logits_s_masked (2, N, k, Nd, L, V)

# === greedy strats ===

class GreedySampleStrategyBase(BidirTrainStrategyBase):
    def __init__(self) -> None:
        super().__init__()

    def recon_proto_and_recon_daughter(strat, self: biDirReconModelRNN | biDirReconModelTrans, batch):
        
        # assumes batch[0] is d2p and batch[1] is from p2d AND they correspond to same cognate set (bug alert)
        
        (daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_seq), _, (prompted_proto_seqs_s, prompted_proto_seqs_lens_s, daughters_ipa_langs_s, daughters_lang_langs_s, daughters_seqs_s) = batch
        
        # prompted_proto_seqs_s (N, Nd, L_p)
        # prompted_proto_seqs_lens_s (N, Nd, 1)
        # daughters_ipa_langs_s (N, Nd, 1)
        # daughters_lang_langs_s (N, Nd, 1)
        # daughters_seqs_s (N, Nd, L_d)
        
        N = daughters_concat_seq.shape[0]
        Nd = daughters_seqs_s.shape[1]
        
        proto_lang_ipa_ids = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        proto_lang_lang_ids = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        
        # === daughter to proto recon ===
        # (no training! only to get a predicted proto)
        
        predicted_proto = self.d2p.greedy_decode(daughters_concat_seq, daughters_concat_langs, daughters_concat_seqs_lens, proto_lang_lang_ids)
        
        proto_predicted_correct = sequences_equal(predicted_proto, proto_seq)
        
        # print('predicted proto', predicted_proto)
        # print('gold proto', proto_seq)
        # print('proto_predicted_correct', proto_predicted_correct)
        
        # === proto to daughter recon ===
        
        # num_daughters = daughters_seqs_s.shape[0]    
        # target_prompted_predicted_proto = torch.cat((daughters_ipa_langs_s, repeat(predicted_proto, '1 L -> Nd L', Nd=num_daughters)), dim=-1)
        
        target_prompted_predicted_proto = torch.cat((daughters_ipa_langs_s, repeat(predicted_proto, 'N L -> N Nd L', Nd=Nd)), dim=-1)
        valid_daughters_mask = (daughters_ipa_langs_s != PAD_IDX) # (N Nd 1)
        
        target_prompted_predicted_proto_lens = torch.sum(target_prompted_predicted_proto != PAD_IDX, dim=2).unsqueeze(2) + (daughters_ipa_langs_s == PAD_IDX)
        
        daughters_lang_langs_s_dummy_for_pad = None if daughters_lang_langs_s == None else rearrange(daughters_lang_langs_s, 'N Nd 1 -> (N Nd) 1') + ((rearrange(daughters_lang_langs_s, 'N Nd 1 -> (N Nd) 1') == PAD_IDX) * self.p2d.min_possible_target_lang_langs_idx)
        
        p2d_decode_res = self.p2d.teacher_forcing_decode(rearrange(target_prompted_predicted_proto, 'N Nd L -> (N Nd) L'), None, rearrange(target_prompted_predicted_proto_lens, 'N Nd 1 -> (N Nd) 1'), rearrange(daughters_seqs_s, 'N Nd L -> (N Nd) L'), daughters_lang_langs_s_dummy_for_pad)
        
        return N, Nd, valid_daughters_mask, daughters_seqs_s, daughters_ipa_langs_s, p2d_decode_res, proto_predicted_correct
