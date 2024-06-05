from __future__ import annotations
from typing import Annotated
from typing import TYPE_CHECKING

from models.biDirReconIntegration import biDirReconModelRNN, biDirReconModelTrans
from models.bidir_strats.base import *
import wandb
from copy import copy
from .consistency_regularisation_lib import sigmoid_rampup, augment, mse_consistency_loss, kl_consistency_loss
someBiDirModel = biDirReconModelRNN | biDirReconModelTrans

class GreedySampleCringeBackpropThroughout(GreedySampleStrategyBase):
    def __init__(strat,
        d2p_recon_loss_weight: float,
        d2p_kl_loss_weight: float, # only useful if using vae latent on d2p
        emb_pred_loss_weight: float,
        p2d_loss_on_gold_weight: float,
        p2d_loss_on_pred_weight: float,
        cringe_alpha: float,
        cringe_k: int,
        alignment_convolution_masking: bool,
        convolution_masking_residue: float,
        
        enable_pi_model: bool,
        pi_consistency_type: str | None, # "mse" | "kl"
        pi_consistency_rampup_length: int | None, # e.g. 5
        pi_max_consistency_scaling: float | None, # e.g. 5.0
        pi_proportion_labelled: float | None, # in [0, 1], corresponds to M/N in laineTemporalEnsemblingSemiSupervised2017

    ) -> None:
        super().__init__()
        strat.d2p_recon_loss_weight = d2p_recon_loss_weight
        strat.d2p_kl_loss_weight = d2p_kl_loss_weight
        strat.emb_pred_loss_weight = emb_pred_loss_weight
        strat.p2d_loss_on_gold_weight = p2d_loss_on_gold_weight
        strat.p2d_loss_on_pred_weight = p2d_loss_on_pred_weight
        
        strat.cringe_alpha = cringe_alpha
        strat.cringe_k = cringe_k
        strat.alignment_convolution_masking = alignment_convolution_masking
        strat.convolution_masking_residue = convolution_masking_residue
        
        strat.enable_emb_pred_pass = strat.emb_pred_loss_weight != 0.0
        strat.enable_p2d_on_gold_pass = strat.p2d_loss_on_gold_weight != 0.0
        strat.enable_p2d_on_pred_pass = strat.p2d_loss_on_pred_weight != 0.0
        
        strat.cross_entropy_loss_fn_noreduce = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')
        strat.cringe_loss_fn = CringeLoss(ignore_index=PAD_IDX, alpha=strat.cringe_alpha, k=strat.cringe_k)
        
        strat.enable_pi_model = enable_pi_model
        if strat.enable_pi_model:
            strat.unsupervised_weight_func = lambda epoch: strat.get_current_consistency_weight(epoch)

            strat.consistency_type = pi_consistency_type
            strat.consistency_rampup_length = pi_consistency_rampup_length
            strat.max_consistency_scaling = pi_max_consistency_scaling
            strat.proportion_labelled = pi_proportion_labelled
            
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
                self.decoder_state2embedding = MLP(
                    input_dim = self.d2p.model_size,
                    feedforward_dim = self.d2p.feedforward_dim, 
                    output_size = self.p2d.embedding_dim,
                ).to(self.device)
            case biDirReconModelTrans():
                self.decoder_state2embedding = MLP(
                    input_dim = self.d2p.model_dim,
                    feedforward_dim = self.d2p.feedforward_dim, 
                    output_size = self.p2d.embedding_dim,
                ).to(self.device)
        
        print("running extra init")
        
        if not self.universal_embedding:
            print("ERROR: universal_embedding required for this strategy.")
            assert False
        
        match self:
            case biDirReconModelRNN():
                assert self.p2d.embeddings == self.d2p.embeddings == self.embeddings
                assert self.p2d.embedding_dim == self.d2p.embedding_dim == self.universal_embedding_dim
                if strat.enable_pi_model:
                    assert self.d2p.use_vae_latent == False # no vae support here
            case biDirReconModelTrans():
                assert self.p2d.ipa_embedding == self.d2p.ipa_embedding == self.ipa_embedding
                assert self.p2d.lang_embedding == self.d2p.lang_embedding == self.lang_embedding
                assert self.p2d.embedding_dim == self.d2p.embedding_dim == self.universal_embedding_dim

        self.samples_tables['val'] = wandb.Table(columns=["epoch", "p", "p_hat", "d_hat_on_pred", "d_hat_on_gold", "d", "d_lang", "pp_corr", "dp_on_pred_corr", "dp_on_gold_corr"])
        self.samples_tables['test'] = wandb.Table(columns=["epoch", "p", "p_hat", "d_hat_on_pred", "d_hat_on_gold", "d", "d_lang", "pp_corr", "dp_on_pred_corr", "dp_on_gold_corr"])
        self.samples_tables['transductive'] = wandb.Table(columns=["epoch", "p", "p_hat", "d_hat_on_pred", "d_hat_on_gold", "d", "d_lang", "pp_corr", "dp_on_pred_corr", "dp_on_gold_corr"])

        return

    def get_current_consistency_weight(strat, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242, implemented by athiwaratkunThereAreMany2019
        if not strat.enable_pi_model:
            raise ValueError("pi model not enabled")
        return strat.max_consistency_scaling * strat.proportion_labelled * sigmoid_rampup(epoch, strat.consistency_rampup_length)
    
    def training_step(strat, self: biDirReconModel, batch: batch_t, batch_idx: int) -> None:
        res_dict = strat.forward(self, batch, batch_idx, 'train')
        return res_dict['loss']
    
    def on_train_epoch_end(strat, self: biDirReconModel) -> dict | None:
        pass
    
    def validation_step(strat, self: biDirReconModel, batch: batch_t, batch_idx: int) -> None:
        res_dict = strat.shared_eval_step(self, batch, batch_idx, 'val')
        return
    
    def test_step(strat, self: biDirReconModel, batch: batch_t, batch_idx: int) -> None:
        if self.transductive_test:
            res_dict = strat.shared_eval_step(self, batch, batch_idx, 'transductive')
        else:
            res_dict = strat.shared_eval_step(self, batch, batch_idx, 'test')
        return
    
    def shared_eval_step(strat, self: biDirReconModel, batch: batch_t, batch_idx: int, prefix: str) -> None:
        res_dict = strat.forward(self, batch, batch_idx, prefix)
        if not 'step_res_dicts' in  self.evaluation_step_outputs:
            self.evaluation_step_outputs['step_res_dicts'] = []
        self.evaluation_step_outputs['step_res_dicts'].append(res_dict)
        return
    
    def forward(strat, self: biDirReconModel, batch, batch_idx, curr_loop_mode):
        train: bool = curr_loop_mode == 'train'
        eval: bool = curr_loop_mode == 'val' or curr_loop_mode == 'test' or curr_loop_mode == 'transductive'
        is_transformer: bool
        match self:
            case biDirReconModelRNN():
                is_transformer = False
            case biDirReconModelTrans():
                is_transformer = True
            case _:
                raise Absurd
            
        # region === prep ===

        # p_l indicates we leak label for transductive evaluation on the hidden labels
        (d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s) = batch
        
        # prompted_p_tkns_s (N, Nd, Lp+2)
        # prompted_p_lens (N, Nd, 1)
        # d_ipa_langs_s (N, Nd, 1)
        # d_lang_langs_s (N, Nd, 1)
        # d_tkns_s (N, Nd, Ld+1)
        # p_tkns (N Lp+1)
        # Lp := len of proto without bos (and without prompt)
        
        N = d_cat_tkns.shape[0]
        Lp = p_tkns.shape[1] - 1
        Nd = d_tkns_s.shape[1]
        Ld = d_tkns_s.shape[-1] - 1
        valid_d_mask = (d_lang_langs_s != PAD_IDX) # (N Nd 1)
        p_padding_mask = (p_tkns == PAD_IDX) # (N Lp+1)
        # p_l_padding_mask = (p_l_tkns == PAD_IDX) # (N Lp+1) (strictly for evaluation! don't leak labels)
        d_padding_mask = (d_tkns_s == PAD_IDX) # (N, Nd, Ld+1)
        # Lp_l = p_l_tkns.shape[1] - 1
        
        from prelude import LabelStatus
        labelled_p_mask = (p_fs == LabelStatus.LABELLED.value) # (N, 1)
        nonlabelled_p_mask = (~ labelled_p_mask) # (N, 1)
        pseudolabelled_p_mask = (p_fs == LabelStatus.PSEUDOLABEL.value) # (N, 1)
        unlabelled_p_mask = (p_fs == LabelStatus.UNLABELLED.value) # (N, 1)
        N_labelled = torch.sum(labelled_p_mask)
        N_nonlabelled = torch.sum(nonlabelled_p_mask)
        N_pseudolabelled = torch.sum(pseudolabelled_p_mask)
        N_unlabelled = torch.sum(unlabelled_p_mask)
        
        p_ipa_lang_vec = repeat((torch.LongTensor([self.ipa_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        p_lang_lang_vec = repeat((torch.LongTensor([self.lang_vocab.get_idx(self.protolang)]).to(self.device)), '1 -> N 1', N=N)
        
        daughters_lang_langs_dummyized_pad_s = d_lang_langs_s + ((d_lang_langs_s == PAD_IDX) * self.p2d.min_possible_target_lang_langs_idx)
        
        d2p_target_langs = p_lang_lang_vec if ((not is_transformer) and self.d2p.lang_embedding_when_decoder) else None

        match self:
            case biDirReconModelRNN():
                transformer_d2p_d_cat_style=False
            case biDirReconModelTrans():
                transformer_d2p_d_cat_style=True

        # no vae support on p2d yet
        assert is_transformer or (self.p2d.use_vae_latent == False)
        
        # endregion
        
        # region === proto recon ===
            
        # 1 > teacher forcing d2p
        
        if not strat.enable_pi_model:
            
            if not is_transformer:
            
                d2p_decode_res = self.d2p.teacher_forcing_decode(d_cat_tkns, d_cat_langs, d_cat_lens, p_tkns, d2p_target_langs)
                
                d2p_logits = d2p_decode_res['logits'] # d2p_logits (N, Lp, V)
                d2p_encoder_states =  d2p_decode_res['encoder_states']
                encoder_h_n =  d2p_decode_res['encoder_h_n']
                d2p_decoder_states =  d2p_decode_res['decoder_states'] # (N, Lp, Dd2p * Hd2p)
                assert d2p_decoder_states.shape[-1] == self.d2p.model_size # (N, Lp, Hd2p), bidirectional decoder not supported yet
                # IDEA: get the attention weighted states out?
                
                _, d2p_recon_loss, d2p_kl_loss = self.d2p.get_loss_from_teacher_forcing_decode_res(d2p_decode_res, p_tkns)
            
            else:
                
                d2p_logits, d2p_recon_loss, d2p_decoder_states = self.d2p.forward_on_batch(batch) # (N, Lp, V)
                d2p_kl_loss = 0.0
            
        else:
            
            # 1 > augment
            
            d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, d_indv_lens_aug1 = augment(d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, transformer_d2p_d_cat_style=transformer_d2p_d_cat_style)
            d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, d_indv_lens_aug2 = augment(d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, transformer_d2p_d_cat_style=transformer_d2p_d_cat_style)

            # 2 > forward on both
                
            match self:
                case biDirReconModelRNN():
                    decode_res_aug1 = self.d2p.teacher_forcing_decode(d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, p_tkns, d2p_target_langs)
                    logits_aug1 = decode_res_aug1['logits'] # (N, Lp, V)
                    decode_res_aug2 = self.d2p.teacher_forcing_decode(d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, p_tkns, d2p_target_langs)
                    logits_aug2 = decode_res_aug2['logits'] # (N, Lp, V)
                    
                    # 4 > adapt
                    
                    d2p_decode_res = decode_res_aug1
                    
                    d2p_logits = d2p_decode_res['logits'] # d2p_logits (N, Lp, V)
                    d2p_encoder_states =  d2p_decode_res['encoder_states']
                    encoder_h_n =  d2p_decode_res['encoder_h_n']
                    d2p_decoder_states =  d2p_decode_res['decoder_states'] # (N, Lp, Dd2p * Hd2p)
                    assert d2p_decoder_states.shape[-1] == self.d2p.model_size # (N, Lp, Hd2p), bidirectional decoder not supported yet

                    d2p_decode_res = decode_res_aug1
                case biDirReconModelTrans():
                    
                    logits_aug1, d2p_recon_loss_aug1, d2p_decoder_states_aug1 = self.d2p.forward_on_batch((
                        (d_cat_tkns_aug1, d_cat_langs_aug1, d_cat_lens_aug1, d_indv_lens_aug1, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                    )) # (N, Lp, V)
                    logits_aug2, d2p_recon_loss_aug2, d2p_decoder_states_aug2 = self.d2p.forward_on_batch((
                        (d_cat_tkns_aug2, d_cat_langs_aug2, d_cat_lens_aug2, d_indv_lens_aug2, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (prompted_p_tkns, prompetd_p_lens, d_ipa_langs, d_lang_langs, d_tkns), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)
                    )) # (N, Lp, V)
                    
                    # 4 > adapt
                    
                    d2p_logits = logits_aug1
                    d2p_decoder_states = d2p_decoder_states_aug1
                
                case _:
                    raise Absurd

            
            # 3 > compute loss
            
            pi_recon_loss = utils.calc_cross_entropy_loss(
                logits_aug1.swapaxes(1,2), # (N, V, T)
                p_tkns[:, 1:] # (N, T)
            )
                
            pi_consistency_loss = strat.consistency_criterion(
                logits_aug1, # source, gets gradient
                logits_aug2  # target, no gradient flow
            )
            
            
            d2p_recon_loss = pi_recon_loss + strat.unsupervised_weight_func(self.current_epoch) * pi_consistency_loss
            d2p_kl_loss = 0.0
            
            if train:
                try:
                    self.log(f"pi_model_module/train_info/softmax_mse_loss", mse_consistency_loss(logits_aug1, logits_aug2))
                    self.log(f"pi_model_module/train_info/softmax_kl_loss", kl_consistency_loss(logits_aug1, logits_aug2))

                    self.log(f"pi_model_module/{curr_loop_mode}/pi_recon_loss", pi_recon_loss)
                    self.log(f"pi_model_module/{curr_loop_mode}/pi_consistency_loss", pi_consistency_loss)
                except MisconfigurationException: pass

        # 2 > make tokens
        
            # pred_p.shape (N Lp)
            # pred_p_s.shape (N Nd Lp)

        pred_p_junkatpad = torch.argmax(d2p_logits, dim=-1) # (N Lp)
        pred_p = pred_p_junkatpad * (~p_padding_mask[:,1:]) + p_padding_mask[:,1:] * PAD_IDX # (N Lp)
        # pred_p_l = pred_p_junkatpad * (~p_l_padding_mask[:,1:][:, :Lp]) + p_l_padding_mask[:,1:][:, :Lp] * PAD_IDX # (N Lp)
        pred_p_s = repeat(pred_p, f'N Lp -> N {Nd} Lp')

        # 3 > check correctness
        
            # proto_correct (N 1)
            # proto_correct_s (N Nd 1)
        
        proto_correct = sequences_equal(pred_p, p_tkns[:, 1:]).unsqueeze(-1) # regardless of labelling status, for training purpose, NOT train-time evaluation! (validation and test sets okay)
        proto_correct_s = repeat(proto_correct, f'N Lp -> N {Nd} Lp')
        
        # endregion 
        
        
        # region === differentiable bridge ===
                        
        # 1 > embedding pred
        
            # pred_p_pred_emb.shape (N Nd Ep2d). Umprompted and without bos

        pred_p_pred_emb = self.decoder_state2embedding(d2p_decoder_states)
        
        emb_cos_sim_loss_masked = torch.zeros(N, Lp).to(self.device)
        if strat.enable_emb_pred_pass:

            # 2 > actual pred
                
                # pred_p.shape (N Lp)
                # pred_p_emb.shape (N Lp Ep2d)

            with torch.no_grad():
                if not is_transformer:
                    pred_p_actual_emb = self.p2d.embeddings(pred_p, None)
                else:
                    pred_p_actual_emb = self.p2d.ipa_embedding(pred_p) # NOTE this is without positional encoding. need to do positional embedding when injecting into transformer
                    

            # 3 > embedding similarity loss

                # emb_cos_sim_loss_masked (N Lp)
            
            cos_sim_loss_fn = torch.nn.CosineEmbeddingLoss(reduction='none')
            emb_cos_sim_loss_unmasked = rearrange(cos_sim_loss_fn(
                rearrange(pred_p_actual_emb, 'N Lp Ep2d -> (N Lp) Ep2d'),
                rearrange(pred_p_pred_emb, 'N Lp Ep2d -> (N Lp) Ep2d'),
                torch.ones((N * Lp)).to(self.device)
            ), '(N Lp) -> N Lp', N=N, Lp=Lp)
            
            emb_cos_sim_loss_masked = emb_cos_sim_loss_unmasked * (~p_padding_mask[:, 1:])
      
        # 4 > attach prompt and bos
        
            # prompt_emb_s.shape (N Nd 1 Ep2d)
            # bos_emb_s.shape (N Nd 1 Ep2d)
            # prompted_pred_p_pred_emb_s.shape (N Nd Lp+2 Ep2d)

        pred_p_pred_emb_s = repeat(pred_p_pred_emb, f'N Lp Ep2d -> N {Nd} Lp Ep2d')
        if not is_transformer:
            prompt_emb_s = self.p2d.embeddings(prompted_p_tkns_s[:,:,0:1], None)
            bos_emb_s = self.p2d.embeddings(prompted_p_tkns_s[:,:,1:2], None)
        else:
            prompt_emb_s = self.p2d.ipa_embedding(prompted_p_tkns_s[:,:,0:1])
            bos_emb_s = self.p2d.ipa_embedding(prompted_p_tkns_s[:,:,1:2])
        prompted_pred_p_pred_emb_s = torch.cat((prompt_emb_s, bos_emb_s, pred_p_pred_emb_s), dim=-2)
        
        # endregion
        
        
        # region === daughter recon on gold p ===
        
        p2d_recon_loss_on_gold_masked_s = torch.zeros(N, Nd, Ld).to(self.device)
        pred_d_on_gold_s = torch.zeros(N, Nd, Ld).to(self.device)
        
        if strat.enable_p2d_on_gold_pass:
                
            # 1 > teacher forcing on p2d
            
            if not is_transformer:
                
                p2d_decode_res_on_gold = self.p2d.teacher_forcing_decode(
                    source_tokens = rearrange(prompted_p_tkns_s, 'N Nd Lpp -> (N Nd) Lpp'),
                    source_langs = None, 
                    source_seqs_lens = rearrange(prompted_p_lens, 'N Nd 1 -> (N Nd) 1'),
                    target_tokens = rearrange(d_tkns_s, 'N Nd Ld -> (N Nd) Ld'), 
                    target_langs = rearrange(daughters_lang_langs_dummyized_pad_s, 'N Nd 1 -> (N Nd) 1'), 
                )
                p2d_logits_on_gold_us = p2d_decode_res_on_gold['logits']

            else:
                
                p2d_logits_on_gold_us, p2d_recon_on_gold_loss, p2d_decoder_on_gold_states = self.p2d.forward_on_batch(
                    ((d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (
                        rearrange(prompted_p_tkns_s, 'N Nd Lpp -> (N Nd) Lpp'), # prompted_p_tkns
                        rearrange(prompted_p_lens, 'N Nd 1 -> (N Nd) 1'), # prompetd_p_lens
                        None, # d_ipa_langs # WARN: set to none because transfomer discard them
                        None, # d_lang_langs # WARN: set to none because transfomer discard them
                        rearrange(d_tkns_s, 'N Nd Ld -> (N Nd) Ld'), # d_tkn
                    ), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s))
                ) # (N, Lp, V)
                
            # 2 > calc loss and mask

            p2d_logits_on_gold_s = rearrange(p2d_logits_on_gold_us, f'(N Nd) Ld V -> N Nd Ld V', N=N, Nd=Nd)
            
            p2d_recon_loss_on_gold_unmasked_s = strat.cross_entropy_loss_fn_noreduce(
                rearrange(p2d_logits_on_gold_s, 'N Nd Ld V -> N V Nd Ld'), 
                d_tkns_s[:,:,1:],
            ) # sth like tensor(5.4489, grad_fn=<NllLoss2DBackward0>)
            p2d_recon_loss_on_gold_masked_s = p2d_recon_loss_on_gold_unmasked_s * valid_d_mask

            # 3 > make tokens
            
            pred_d_on_gold_s_junkatpad = torch.argmax(p2d_logits_on_gold_s, dim=-1) # (N Nd Ld)
            pred_d_on_gold_s = pred_d_on_gold_s_junkatpad * (~d_padding_mask[:,:,1:]) + d_padding_mask[:,:,1:] * PAD_IDX


        # endregion
        
        
        # region === daughter recon on predicted p === 
        
        pred_d_on_pred_s = torch.zeros(N, Nd, Ld).to(self.device)
        p2d_loss_on_pred_s = torch.zeros(N, Nd).to(self.device)
        p2d_ce_loss_on_pred_s = torch.zeros(N, Nd).to(self.device)
        p2d_cr_loss_on_pred_s = torch.zeros(N, Nd).to(self.device)
        
        if strat.enable_p2d_on_pred_pass:
        
            # 1 > teacher forcing on p2d
            
                # prompted_proto_seqs_s.shape (N Nd Lp+2)
                # prompted_proto_seqs_lens_s.shape (N Nd 1)
                # daughters_ipa_langs_s.shape (N Nd 1)
                # daughters_lang_langs_s.shape (N Nd 1)
                # daughters_lang_langs_dummyized_pad_s (N Nd 1)
                # daughters_seqs_s.shape (N Nd Ld)
                # prompted_pred_p_pred_emb_s.shape (N Nd Lp+2 Ep2d)

            if not is_transformer:

                p2d_decode_res_on_pred = self.p2d.teacher_forcing_decode(
                    source_tokens = torch.zeros_like(rearrange(prompted_p_tkns_s, 'N Nd Lpp -> (N Nd) Lpp')), # doesn't matter
                    source_langs = None, # doesn't matter
                    source_seqs_lens = rearrange(prompted_p_lens, 'N Nd 1 -> (N Nd) 1'),
                    target_tokens = rearrange(d_tkns_s, 'N Nd Ld -> (N Nd) Ld'), 
                    target_langs = rearrange(daughters_lang_langs_dummyized_pad_s, 'N Nd 1 -> (N Nd) 1'), 
                    inject_embedding = rearrange(prompted_pred_p_pred_emb_s, 'N Nd Lpp Ep2d -> (N Nd) Lpp Ep2d'),
                )
                p2d_logits_on_pred_us = p2d_decode_res_on_pred['logits']
            
            else:

                embedding_injection = self.p2d.pos_encoding(
                    rearrange(prompted_pred_p_pred_emb_s, 'N Nd Lpp Ep2d -> (N Nd) Lpp Ep2d'),
                    (torch.ones(N * Nd, 1) * Lp+2).long()
                )
                p2d_logits_on_pred_us, p2d_recon_on_pred_loss, p2d_decoder_on_pred_states = self.p2d.forward_on_batch(
                    ((d_cat_tkns, d_cat_langs, d_cat_lens, d_indv_lens, p_lang_lang_vec, p_tkns, p_l_tkns, p_fs), (
                        torch.zeros_like(rearrange(prompted_p_tkns_s, 'N Nd Lpp -> (N Nd) Lpp')), # prompted_p_tkns, doesn't matter
                        rearrange(prompted_p_lens, 'N Nd 1 -> (N Nd) 1'), # prompetd_p_lens
                        None, # d_ipa_langs # WARN: set to none because transfomer discard them
                        None, # d_lang_langs # WARN: set to none because transfomer discard them
                        rearrange(d_tkns_s, 'N Nd Ld -> (N Nd) Ld'), # d_tkn
                    ), (prompted_p_tkns_s, prompted_p_lens, d_ipa_langs_s, d_lang_langs_s, d_tkns_s)),
                    inject_embedding = embedding_injection,
                ) # (N, Lp, V)

            # 2 > calc loss and mask
            
                # p2d_logits_us ((N Nd) Ld V)
                # p2d_logits_s (N Nd Ld V)
                # valid_d_mask (N Nd 1)
                # p2d_recon_loss_unmasked_s (N Nd Ld)
                # p2d_recon_loss_masked_s (N Nd Ld)
            
            p2d_logits_on_pred_s = rearrange(p2d_logits_on_pred_us, f'(N Nd) Ld V -> N Nd Ld V', N=N, Nd=Nd)
                    
            # 3 > make tokens
            
            pred_d_on_pred_s_junkatpad = torch.argmax(p2d_logits_on_pred_s, dim=-1) # (N Nd Ld)
            pred_d_on_pred_s = pred_d_on_pred_s_junkatpad * (~d_padding_mask[:,:,1:]) + d_padding_mask[:,:,1:] * PAD_IDX
            
            # 4 > CRINGE loss
            

            _p2d_loss_on_pred_s, p2d_ce_loss_on_pred_s, p2d_cr_loss_on_pred_s = map(
                lambda l_t: rearrange(l_t, '(N Nd Ld) -> N Nd Ld', N=N, Nd=Nd, Ld=p2d_logits_on_pred_s.shape[-2]) * valid_d_mask, 
                strat.cringe_loss_fn(
                    rearrange(p2d_logits_on_pred_s, 'N Nd Ld V -> (N Nd Ld) V'), 
                    rearrange(d_tkns_s[:,:,1:], 'N Nd Ld -> (N Nd Ld)'),
                    rearrange(
                        repeat(proto_correct_s.long(), 'N Nd 1 -> N Nd Ld', Ld=p2d_logits_on_pred_s.shape[-2]), 
                    'N Nd Ld -> (N Nd Ld)')
                )
            ) # ((N Nd Ld), (N Nd Ld), (N Nd Ld))
            
            # 5 > Convolutional masking NOT PLANNED
            
            if strat.alignment_convolution_masking:
            
                unprompted_unBOS_proto_diff = (prompted_p_tkns_s[:, :, 2:] != pred_p_s).float() # (N Nd Lp); 1 if proto diff, 0 if proto same, disregard target lang prompt and BOS
                
                convolution_mask = rearrange(F.conv1d(
                    rearrange(unprompted_unBOS_proto_diff, 'N Nd Lp -> (N Nd) 1 Lp'), 
                    torch.tensor([[[0.05, 0.1, 0.35, 0.35, 0.1, 0.05]]]).to(self.device), 
                    bias=None, stride=1, padding=2, dilation=1, groups=1
                ), '(N Nd) 1 L -> N Nd L', N=N, Nd=Nd) # (N, Nd, L_target-sth)
                
                if convolution_mask.shape[-1] < p2d_cr_loss_on_pred_s.shape[-1]:
                    convolution_mask = F.pad(convolution_mask, (0, p2d_cr_loss_on_pred_s.shape[-1] - convolution_mask.shape[-1]))
                convolution_mask_for_daughter_len = convolution_mask[:,:,:p2d_cr_loss_on_pred_s.shape[-1]]
                assert (convolution_mask_for_daughter_len.shape == p2d_cr_loss_on_pred_s.shape) # (N, Nd, L_target-sth)
                
                convolution_masked_p2d_cr_loss_on_pred_s = p2d_cr_loss_on_pred_s * strat.convolution_masking_residue + p2d_cr_loss_on_pred_s * convolution_mask_for_daughter_len
                    
                p2d_cr_loss_on_pred_s = convolution_masked_p2d_cr_loss_on_pred_s
                
            p2d_loss_on_pred_s = p2d_ce_loss_on_pred_s + strat.cringe_loss_fn.alpha * p2d_cr_loss_on_pred_s
            
        # endregion
                
                
        # region === conclude and optimise ===
        
        p2d_loss_on_gold = torch.mean(p2d_recon_loss_on_gold_masked_s)
        p2d_loss_on_pred = torch.mean(p2d_loss_on_pred_s)
        emb_pred_loss = torch.mean(emb_cos_sim_loss_masked)
        
        loss = \
            + strat.d2p_recon_loss_weight * d2p_recon_loss \
            + strat.d2p_kl_loss_weight * d2p_kl_loss \
            + strat.p2d_loss_on_gold_weight * p2d_loss_on_gold \
            + strat.p2d_loss_on_pred_weight * p2d_loss_on_pred \
            + strat.emb_pred_loss_weight * emb_pred_loss
            
        
        # endregion
        
        
        # region === train time metrics ===
        
            
        # 1 > proto (Noop)
        
                
        
        # 2 > daughter on gold proto
                
        n_valid_d = torch.sum(valid_d_mask)
        
        correct_d_mask_on_gold = sequences_equal(pred_d_on_gold_s, d_tkns_s[:,:,1:]).unsqueeze(-1) * valid_d_mask # (N Nd 1)
        d_accuracy_on_gold = torch.sum(correct_d_mask_on_gold) / n_valid_d
        
        # 3 > daughter on pred proto
                
        correct_d_mask_on_pred = sequences_equal(pred_d_on_pred_s, d_tkns_s[:,:,1:]).unsqueeze(-1) * valid_d_mask
        
        d_corr_on_pred_p_corr = correct_d_mask_on_pred * proto_correct_s * valid_d_mask
        d_inco_on_pred_p_corr = ~correct_d_mask_on_pred * proto_correct_s * valid_d_mask
        d_inco_on_pred_p_inco = ~correct_d_mask_on_pred * ~proto_correct_s * valid_d_mask
        d_corr_on_pred_p_inco = correct_d_mask_on_pred * ~proto_correct_s * valid_d_mask
        
        d_corr_on_pred_p_corr_rate = torch.sum(d_corr_on_pred_p_corr) / n_valid_d
        d_inco_on_pred_p_corr_rate = torch.sum(d_inco_on_pred_p_corr) / n_valid_d
        d_inco_on_pred_p_inco_rate = torch.sum(d_inco_on_pred_p_inco) / n_valid_d
        d_corr_on_pred_p_inco_rate = torch.sum(d_corr_on_pred_p_inco) / n_valid_d

        # 4 > daughter consistency with emb pred gold or tkn gold
                    
        d_on_gold_d_on_pred_p_corr_consistent_mask = sequences_equal(pred_d_on_pred_s, pred_d_on_gold_s).unsqueeze(-1) * proto_correct_s * valid_d_mask
        
        d_on_gold_d_on_pred_p_corr_consistency = torch.sum(d_on_gold_d_on_pred_p_corr_consistent_mask) / torch.sum(proto_correct_s * valid_d_mask)

        # endregion


        # region === logging ===
        
        res_dict = {
            "N": N,
            "Nd": Nd,
            "n_valid_d": n_valid_d,
            
            "d2p_recon_loss": d2p_recon_loss,
            "d2p_kl_loss": d2p_kl_loss,
            "p2d_loss_on_gold": p2d_loss_on_gold,
            "p2d_loss_on_pred": p2d_loss_on_pred,
            "p2d_ce_loss_on_pred_s": torch.mean(p2d_ce_loss_on_pred_s),
            "p2d_cr_loss_on_pred_s": torch.mean(p2d_cr_loss_on_pred_s),
            "emb_pred_loss": emb_pred_loss,
            "loss": loss,

            # "labelled_p_acc": labelled_p_acc,
            # "nonlabelled_p_acc": nonlabelled_p_acc,
            # "pseudolabelled_p_acc": pseudolabelled_p_acc,
            # "unlabelled_p_acc": unlabelled_p_acc,
            
            "d_accuracy_on_gold": d_accuracy_on_gold,
            "d_corr_on_pred_p_corr_rate": d_corr_on_pred_p_corr_rate,
            "d_inco_on_pred_p_corr_rate": d_inco_on_pred_p_corr_rate,
            "d_inco_on_pred_p_inco_rate": d_inco_on_pred_p_inco_rate,
            "d_corr_on_pred_p_inco_rate": d_corr_on_pred_p_inco_rate,
            "d_on_gold_d_on_pred_p_corr_consistency": d_on_gold_d_on_pred_p_corr_consistency,
        }
        
        if train:
            try:
                self.log(f"combined/train/lr", self.optimizer.param_groups[0]['lr'], prog_bar=True)
                for k in [
                    "d2p_recon_loss", 
                    "p2d_loss_on_gold",
                    "p2d_loss_on_pred",
                    "p2d_ce_loss_on_pred_s",
                    "p2d_cr_loss_on_pred_s",
                    "emb_pred_loss",
                    "d_accuracy_on_gold",
                    "d_corr_on_pred_p_corr_rate",
                    "d_inco_on_pred_p_corr_rate",
                    "d_inco_on_pred_p_inco_rate",
                    "d_corr_on_pred_p_inco_rate",
                    "d_on_gold_d_on_pred_p_corr_consistency"
                ]: 
                    self.log(f"combined/train/{k}", res_dict[k])
                    
                self.log(f"combined/train/loss", res_dict['loss'], prog_bar=True)
                
    
            except MisconfigurationException: pass
        
        if eval:
            try:
                for k in [ # these are average over batches, so weighted by batch size
                    "d2p_recon_loss", 
                    "loss",
                ]: 
                    self.log(f"combined/{curr_loop_mode}/{k}", res_dict[k], on_step=False, on_epoch=True, batch_size=N)
                

                for k in [ # these are average weighted by number of valid daughters
                    "p2d_loss_on_gold",
                    "p2d_loss_on_pred",
                    "p2d_ce_loss_on_pred_s",
                    "p2d_cr_loss_on_pred_s",
                    "emb_pred_loss",
                    "d_accuracy_on_gold",
                    "d_corr_on_pred_p_corr_rate",
                    "d_inco_on_pred_p_corr_rate",
                    "d_inco_on_pred_p_inco_rate",
                    "d_corr_on_pred_p_inco_rate",
                    "d_on_gold_d_on_pred_p_corr_consistency"
                ]: 
                    self.log(f"combined/{curr_loop_mode}/{k}", res_dict[k], on_step=False, on_epoch=True, batch_size=int(n_valid_d.item()))
            except MisconfigurationException: pass

            # pred_p_s (N Nd Lp) !
            # proto_seq (N Lp+1) !
            # pred_d_s (N Nd Ld) !
            # daughters_seqs_s (N Nd Ld) !
            # valid_d_mask (N Nd 1) !
            
            res_dict['proto_seq'] = p_tkns # (N Lp+1)
            res_dict['pred_p_s'] = pred_p_s # already padded `* (~repeat(p_padding_mask[:,1:], 'N L -> N Nd L', Nd=Nd))`? # (N Nd Lp)
            res_dict['pred_d_on_gold_s'] = pred_d_on_gold_s # (N Nd Ld)
            res_dict['pred_d_on_pred_s'] = pred_d_on_pred_s # (N Nd Ld)
            res_dict['daughters_seqs_s'] = d_tkns_s # (N Nd Ld)
            res_dict['daughters_ipa_langs_s'] = d_ipa_langs_s # (N Nd 1)
            res_dict['valid_d_mask'] = valid_d_mask # (N Nd 1)
        
        # endregion
            
        return res_dict

    def on_validation_epoch_end(strat, self: biDirReconModel):
        return strat.shared_eval_epoch_end(self, 'val')
    
    def on_test_epoch_end(strat, self: biDirReconModel):
        if self.transductive_test:
            return strat.shared_eval_epoch_end(self, 'transductive')
        else:
            return strat.shared_eval_epoch_end(self, 'test')

    def shared_eval_epoch_end(strat, self: biDirReconModel, prefix: str) -> dict | None:
          
        # 1 > crunch val steps data
        
        for step_res_dict in self.evaluation_step_outputs['step_res_dicts']:
            N = step_res_dict['N']
            Nd = step_res_dict['Nd']

            proto_seq = step_res_dict['proto_seq']
            pred_p_s = step_res_dict['pred_p_s']
            pred_d_on_gold_s = step_res_dict['pred_d_on_gold_s']
            pred_d_on_pred_s = step_res_dict['pred_d_on_pred_s']
            daughters_seqs_s = step_res_dict['daughters_seqs_s']
            valid_d_mask = step_res_dict['valid_d_mask']
            daughters_ipa_langs_s = step_res_dict['daughters_ipa_langs_s']
            
            def pad_str(str_, len_):
                return str_ + " "*(max(0, len_ - len(str_)))
        
            # print some samples
            for n in range(N):
                for nd in range(Nd):
                    if valid_d_mask[n][nd]:
                        p = ''.join(self.d2p.ipa_vocab.to_tokens(proto_seq[n]))
                        pp = ''.join(self.d2p.ipa_vocab.to_tokens(pred_p_s[n][nd]))
                        
                        d = ''.join(self.p2d.ipa_vocab.to_tokens(daughters_seqs_s[n][nd]))
                        dp_on_gold = ''.join(self.p2d.ipa_vocab.to_tokens(pred_d_on_gold_s[n][nd]))
                        dp_on_pred = ''.join(self.p2d.ipa_vocab.to_tokens(pred_d_on_pred_s[n][nd]))
                        
                        d_lang = ''.join(self.p2d.ipa_vocab.to_tokens(daughters_ipa_langs_s[n][nd], remove_special=False))
                        pp_corr = p == pp
                        dp_on_pred_corr = dp_on_pred == d
                        dp_on_gold_corr = dp_on_gold == d
                        
                        assert isinstance(self.samples_tables[prefix], wandb.Table)
                        self.samples_tables[prefix].add_data(self.current_epoch, p, pp, dp_on_pred, dp_on_gold, d, d_lang, pp_corr, dp_on_pred_corr, dp_on_gold_corr) 
                
        self.evaluation_step_outputs.clear()
        
        # 2 > table logging
        
        if self.current_epoch > 0 and isinstance(self.logger, WandbLogger):
            # self.logger.experiment.log({f'combined/{prefix}/samples': copy(self.samples_tables[prefix])}) # copy hack to workaround https://github.com/wandb/wandb/issues/2981
            pass # TODO unblock table logging when needed, else keeping all tables takes too much space

        # 2 > run eval of submodels?
        
        tmp_evaluator = pl.Trainer(
            accelerator=self.trainer.set_accelerator, 
            enable_progress_bar=False,
            devices=self.trainer.set_devices
        )
        
        match prefix:
            case 'val':
                d2p_eval_res = tmp_evaluator.validate(self.d2p, dataloaders=self.trainer.validate_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
                p2d_eval_res = tmp_evaluator.validate(self.p2d, dataloaders=self.trainer.validate_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
            case 'test':
                d2p_eval_res = tmp_evaluator.test(self.d2p, dataloaders=self.trainer.test_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
                p2d_eval_res = tmp_evaluator.test(self.p2d, dataloaders=self.trainer.test_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
            case 'transductive':
                self.d2p.transductive_test = True
                d2p_eval_res = tmp_evaluator.test(self.d2p, dataloaders=self.trainer.test_loop._data_source.instance, verbose=False)[0]
                self.to(self.device)
                self.d2p.transductive_test = False
                p2d_eval_res = None
            case _:
                raise ValueError(f'prefix {prefix} not supported')
        
        # 3 > logging
               
        try:
            self.log_dict(d2p_eval_res)
            if p2d_eval_res is not None:
                self.log_dict(p2d_eval_res)
        except MisconfigurationException: pass
        
        if p2d_eval_res is not None:
            return {**d2p_eval_res, **p2d_eval_res}
        else:
            return d2p_eval_res