from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from lib.dataset import LabelStatus
import models.utils as utils
from tqdm import tqdm 
import torch
from models.encoderDecoderRNN import Seq2SeqRNN
from models.encoderDecoderTransformer import EncoderDecoderTransformer

class Pseudolabeller(Callback):
    
    def __init__(self, 
        min_epoch: int, # minimum epoch to start bootstrapping
        log_prob_thresh: float, # minimum log probability to be considered a pseudolabel
        pseudolabelling_cap: int, # max number of pseudolabels to make every call
        inference_batch_size: int, # batch size for pseudolabelling
        alpha: float, # noramlization constant for log prob
        
    ):
        self.min_epoch = min_epoch
        self.log_prob_thresh = log_prob_thresh
        self.pseudolabelling_cap = pseudolabelling_cap
        self.inference_batch_size = inference_batch_size
        self.alpha = alpha
    
    def on_train_epoch_end(self, trainer, wrapped_model):
        if trainer.current_epoch < self.min_epoch:
            return
        
        model = wrapped_model.d2p
        
        print("Training epoch ended, attempt to pseudolabel!")
        model.eval()
        assert not model.training
        
        
        # 1 > set up
        
        N = self.inference_batch_size
        train_set = trainer.train_dataloader.dataset
        in_order_train_loader = DataLoader(
            train_set, 
            collate_fn=train_set.collate_fn, 
            batch_size=N,
            shuffle=False,
        )
        
        # 2 > get predictions
        
        all_pred_entries = [] # an entry is (pred, norm_log_prob, index)

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(in_order_train_loader), desc="getting predictions for train set", leave=False, total=len(in_order_train_loader)):
                
                match model:
                    case Seq2SeqRNN():
                        source_tokens, source_langs, source_seqs_lens, target_tokens, target_lang_ipa_ids, target_lang_lang_ids = model.unpack_batch(batch)
                        source_tokens, source_langs, source_seqs_lens, target_lang_lang_ids = source_tokens.to(model.device), source_langs.to(model.device), source_seqs_lens.to(model.device), target_lang_lang_ids.to(model.device)
                        
                        predictions = model.greedy_decode(source_tokens, source_langs, source_seqs_lens, target_lang_lang_ids) # (N, Lpred)
                        
                        (log_probs_seq, log_prob_sum, logits) = model.get_sequence_log_probs(source_tokens, source_langs, source_seqs_lens, predictions, target_lang_lang_ids)
                        # log_prob_sum (N)
                    case EncoderDecoderTransformer():
                        N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(batch, model.device, model.task, model.ipa_vocab, model.lang_vocab, model.protolang)
                        
                        predictions = model.greedy_decode(s_tkns.to(model.device), s_indv_lens.to(model.device), s_langs.to(model.device), s_mask.to(model.device), s_pad_mask.to(model.device), decode_max_len=model.inference_decode_max_length)  # (N, Lpred)
                        
                        (log_probs_seq, log_prob_sum, logits) = model.get_sequence_log_probs(batch, predictions)
                        # log_prob_sum (N)

                
                assert(len(predictions) == len(log_prob_sum))
                for i in range(len(predictions)):
                    pred_tkns = model.ipa_vocab.to_tokens(predictions[i], remove_special=True)
                    pred_len = len(pred_tkns)
                    pred_log_prob = log_prob_sum[i].item()
                    pred_normalized_log_prob = pred_log_prob / ((float(max(pred_len - 1, 0) + 1e-9)) ** self.alpha)
                    
                    all_pred_entries.append((pred_tkns, pred_normalized_log_prob, batch_idx * N + i))
        
        # 3 > sort predictions by norm log prob, highest log prob on top.
        
        all_pred_entries.sort(key=lambda entry: entry[1], reverse=False)
        
        # 4 > keep selecting top predictions, checking if it passes thresh, and filling unlabelled slots if possible
        
        num_updated = 0
        correctly_pseudolabelled = 0
        for (pred_tkns, pred_normalized_log_prob, index) in tqdm(all_pred_entries, total=len(all_pred_entries), desc="pseudolabelling...", leave=False):
            if pred_normalized_log_prob < self.log_prob_thresh: 
                continue
                
            if train_set.Pf[index] != LabelStatus.UNLABELLED:
                continue

            # print(index, train_set.Ps[index][train_set.protolang], train_set.Pl[index][train_set.protolang], train_set.Pf[index])
            train_set.Ps[index][train_set.protolang] = pred_tkns
            train_set.Pf[index] = LabelStatus.PSEUDOLABEL
            # print(index, train_set.Ps[index][train_set.protolang], train_set.Pl[index][train_set.protolang], train_set.Pf[index])
            # print()
            
            if pred_tkns == train_set.Pl[index][train_set.protolang]:
                correctly_pseudolabelled += 1
            
            num_updated += 1

            if num_updated >= self.pseudolabelling_cap:
                break
        
        print(f"pseudolabelled {num_updated}, {correctly_pseudolabelled} of which were correct")
            
        model.train()
            
        return