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

class SupervisedOnlyD2P(BidirTrainStrategyBase):
    def __init__(
        strat,
    ) -> None:
        super().__init__()
    
    def extra_init(strat, self: someBiDirModel):
        self.logger_prefix = 'd2p_supervised_only'

    def training_step(strat, self: someBiDirModel, batch: batch_t, batch_idx: int) -> Tensor:
        
        match self:
            case biDirReconModelRNN():
                _logits, loss, recon_loss, kl_loss = self.d2p.forward_on_batch(batch)
                
            case biDirReconModelTrans():
                _logits, recon_loss, _decoder_out = self.d2p.forward_on_batch(batch)
                loss = recon_loss
                kl_loss = None
                
            case _:
                assert False, 'bad'
            
        # print(recon_loss)
        
        try:
            self.log(f"{self.logger_prefix}/train/loss", loss, prog_bar=True)
            self.log(f"{self.logger_prefix}/train/recon_loss", recon_loss)
            if kl_loss != None:
                self.log(f"{self.logger_prefix}/train/kl_loss", kl_loss)
            self.log(f"{self.logger_prefix}/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
        except MisconfigurationException: pass
                         
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
    