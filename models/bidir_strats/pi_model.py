from __future__ import annotations
from typing import Annotated
from typing import TYPE_CHECKING
from models.biDirReconIntegration import biDirReconModelRNN, biDirReconModelTrans
from models.bidir_strats.base import *
import wandb
from copy import copy
import random
from torch import Tensor
import numpy as np

someBiDirModel = biDirReconModelRNN | biDirReconModelTrans

from .consistency_regularisation_lib import sigmoid_rampup, augment, mse_consistency_loss, kl_consistency_loss

# a strategy that doesn't really use the p2d part
# reference @athiwaratkunThereAreMany2019 https://github.com/benathi/fastswa-semi-sup
class PiModelD2P(BidirTrainStrategyBase):
    def __init__(
        strat,
        consistency_type: str, # "mse" | "kl"
        consistency_rampup_length: int, # e.g. 5
        
        max_consistency_scaling: float, # e.g. 5.0
        proportion_labelled: float, # in [0, 1], corresponds to M/N in laineTemporalEnsemblingSemiSupervised2017
    ) -> None:
        super().__init__()
        strat.unsupervised_weight_func = lambda epoch: strat.get_current_consistency_weight(epoch)
        
        strat.consistency_type = consistency_type
        strat.consistency_rampup_length = consistency_rampup_length
        strat.max_consistency_scaling = max_consistency_scaling
        strat.proportion_labelled = proportion_labelled
        
        match strat.consistency_type:
            case 'mse':
                strat.consistency_criterion = mse_consistency_loss
            case 'kl':
                strat.consistency_criterion = kl_consistency_loss
            case _:
                raise ValueError(f"Unknown consistency type: {strat.consistency_type}")
    
    def extra_init(strat, self: someBiDirModel):
        match self:
            case biDirReconModelRNN():
                assert self.d2p.use_vae_latent == False # no vae support here
            case biDirReconModelTrans():
                pass
            
        self.logger_prefix = 'd2p_pi_model'

    def get_current_consistency_weight(strat, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242, implemented by athiwaratkunThereAreMany2019
        return strat.max_consistency_scaling * strat.proportion_labelled * sigmoid_rampup(epoch, strat.consistency_rampup_length)

    def training_step(strat, self: someBiDirModel, batch: batch_t, batch_idx: int) -> Tensor:
        (d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s) = batch

        # 0 > prep

        N = d_cat_tkns.shape[0]
        
        match self:
            case biDirReconModelRNN():
                d2p_target_langs = p_lang_lang_vec if self.d2p.lang_embedding_when_decoder else None
                transformer_d2p_d_cat_style=False
            case biDirReconModelTrans():
                transformer_d2p_d_cat_style=True
        
        # 1 > augment
        
        d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, d_indv_lens_aug1 = augment(d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, transformer_d2p_d_cat_style=transformer_d2p_d_cat_style)
        d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, d_indv_lens_aug2 = augment(d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, transformer_d2p_d_cat_style=transformer_d2p_d_cat_style)
        
        # 2 > forward on both
        
        match self:
            case biDirReconModelRNN():
                logits_aug1, _, _, _ = self.d2p.forward_on_batch((
                    (d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, d_indv_lens_aug1, d2p_target_langs, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                )) # (N, Lp, V)
                logits_aug2, _, _, _ = self.d2p.forward_on_batch((
                    (d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, d_indv_lens_aug2, d2p_target_langs, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                )) # (N, Lp, V)

                # decode_res_aug1 = self.d2p.teacher_forcing_decode(d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, p_tkns, d2p_target_langs)
                # logits_aug1 = decode_res_aug1['logits'] 

                # decode_res_aug2 = self.d2p.teacher_forcing_decode(d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, p_tkns, d2p_target_langs)
                # logits_aug2 = decode_res_aug2['logits'] # (N, Lp, V)

            case biDirReconModelTrans():
                logits_aug1, _loss, _decoder_out = self.d2p.forward_on_batch((
                    (d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, d_indv_lens_aug1, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                )) # (N, Lp, V)
                logits_aug2, _loss, _decoder_out = self.d2p.forward_on_batch((
                    (d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, d_indv_lens_aug2, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                )) # (N, Lp, V)
            
            case _:
                raise Absurd
        
        # 3 > compute loss
        
        recon_loss = utils.calc_cross_entropy_loss(
            logits_aug1.swapaxes(1,2), # (N, V, T)
            p_tkns[:, 1:] # (N, T)
        )
        
        consistency_loss = strat.consistency_criterion(
            logits_aug1, # source, gets gradient
            logits_aug2  # target, no gradient flow
        )
        
        loss = recon_loss + strat.unsupervised_weight_func(self.current_epoch) * consistency_loss
        
        self.log(f"{self.logger_prefix}/train_info/softmax_mse_loss", mse_consistency_loss(logits_aug1, logits_aug2))
        self.log(f"{self.logger_prefix}/train_info/softmax_kl_loss", kl_consistency_loss(logits_aug1, logits_aug2))
        
        self.log(f"{self.logger_prefix}/train/recon_loss", recon_loss)
        self.log(f"{self.logger_prefix}/train/consistency_loss", consistency_loss)
        self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
        self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(strat, self: someBiDirModel) -> dict | None:
        return
    
    def validation_step(strat, self: someBiDirModel, batch: batch_t, batch_idx: int) -> Tensor:
        return
    
    def test_step(strat, self: someBiDirModel, batch: batch_t, batch_idx: int) -> Tensor:
        return

    def on_validation_epoch_end(strat, self: someBiDirModel):
        return strat.shared_eval_epoch_end(self, 'val')
    
    def on_test_epoch_end(strat, self: someBiDirModel):
        if self.transductive_test:
            return strat.shared_eval_epoch_end(self, 'transductive')
        else:
            return strat.shared_eval_epoch_end(self, 'test')

    def shared_eval_epoch_end(strat, self: someBiDirModel, prefix: str) -> dict | None:
          
        # run d2p's eval routine
        
        tmp_evaluator = pl.Trainer(
            accelerator=self.trainer.set_accelerator, 
            enable_progress_bar=False,
            devices=self.trainer.set_devices
        )
                
        match prefix:
            case 'val':
                d2p_eval_res = tmp_evaluator.validate(self.d2p, dataloaders=self.trainer.validate_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
            case 'test':
                d2p_eval_res = tmp_evaluator.test(self.d2p, dataloaders=self.trainer.test_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
            case 'transductive':
                self.d2p.transductive_test = True
                d2p_eval_res = tmp_evaluator.test(self.d2p, dataloaders=self.trainer.test_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
                self.d2p.transductive_test = False
            case _:
                raise ValueError(f'prefix {prefix} not supported')
        
        # 3 > logging
               
        try:
            self.log_dict(d2p_eval_res)
        except MisconfigurationException: pass
        
        return {**d2p_eval_res}
    