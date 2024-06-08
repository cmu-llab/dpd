PROPORTION_LABELLED="1.0"
DATASET_SEED="2706283079"
GROUP_NAME="group1"

# wikihan_GRU-ΠM
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=GRU --batch_size=256 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.24039385556739204 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=128 --d2p_num_encoder_layers=2 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0006592816075596439 --max_epochs=283 --min_daughters=1 --p2d_all_lang_summary_only=True --pi_consistency_rampup_length=4 --pi_consistency_type=mse --pi_max_consistency_scaling=257.431725811742 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=4 --weight_decay=0

# wikihan_GRU-DPD
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=GRU --batch_size=64 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.3173917190368629 --cringe_k=5 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.23461573361022817 --d2p_embedding_dim=256 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=64 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.8199604411948759 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.3285448537223006 --eps=1e-08 --lr=0.0005142437763826597 --max_epochs=269 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_decode_mode=greedy_search --p2d_dropout_p=0.1691985760847986 --p2d_embedding_dim=384 --p2d_feedforward_dim=512 --p2d_gated_mlp_by_target_lang=False --p2d_inference_decode_max_length=15 --p2d_lang_embedding_when_decoder=False --p2d_loss_on_gold_weight=0.5769897457744835 --p2d_loss_on_pred_weight=0.8716887573779297 --p2d_model_size=128 --p2d_num_encoder_layers=2 --p2d_prompt_mlp_with_one_hot_lang=False --p2d_use_bidirectional_encoder=False --p2d_use_vae_latent=False --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=128 --use_xavier_init=True --warmup_epochs=7 --weight_decay=0

# wikihan_GRU-DPD-ΠM
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=GRU --batch_size=256 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.263776414532768 --cringe_k=3 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.17613286211339285 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=128 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.9313972776119755 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.5967162545212494 --eps=1e-08 --lr=0.0008696220713116625 --max_epochs=275 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_decode_mode=greedy_search --p2d_dropout_p=0.3487502084117227 --p2d_embedding_dim=256 --p2d_feedforward_dim=512 --p2d_gated_mlp_by_target_lang=False --p2d_inference_decode_max_length=15 --p2d_lang_embedding_when_decoder=False --p2d_loss_on_gold_weight=0.3550411760367638 --p2d_loss_on_pred_weight=0.6495407164097999 --p2d_model_size=64 --p2d_num_encoder_layers=2 --p2d_prompt_mlp_with_one_hot_lang=False --p2d_use_bidirectional_encoder=False --p2d_use_vae_latent=False --pi_consistency_rampup_length=22 --pi_consistency_type=mse --pi_max_consistency_scaling=156.38864601770678 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=128 --use_xavier_init=True --warmup_epochs=4 --weight_decay=0

# wikihan_Trans-ΠM
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=Transformer --batch_size=128 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --d2p_dropout_p=0.1685347649735921 --d2p_embedding_dim=256 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0007581263868728475 --max_epochs=382 --min_daughters=1 --p2d_all_lang_summary_only=True --pi_consistency_rampup_length=14 --pi_consistency_type=mse --pi_max_consistency_scaling=200.21562252876905 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=22 --weight_decay=0

# wikihan_Trans-DPD
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=Transformer --batch_size=128 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.0019169125110127805 --cringe_k=4 --d2p_dropout_p=0.2934991724342204 --d2p_embedding_dim=384 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.5601663278208462 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.4404599704857638 --eps=1e-08 --lr=0.0007198915672709616 --max_epochs=232 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.3083452047198607 --p2d_embedding_dim=384 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=15 --p2d_loss_on_gold_weight=0.5421093135701862 --p2d_loss_on_pred_weight=1.0092242454629332 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=128 --use_xavier_init=True --warmup_epochs=40 --weight_decay=0

# wikihan_Trans-DPD-ΠM
python exp.py --logmodel --tags paper $GROUP_NAME exclude_unlabelled --exclude_unlabelled --vram_thresh 2000 --architecture=Transformer --batch_size=64 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.2851215022368949 --cringe_k=4 --d2p_dropout_p=0.3055495927512927 --d2p_embedding_dim=384 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=1.4265390125295534 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.5442262053769168 --eps=1e-08 --lr=0.0007996949451259284 --max_epochs=315 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.18043361588750984 --p2d_embedding_dim=384 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=15 --p2d_loss_on_gold_weight=0.41337061014191906 --p2d_loss_on_pred_weight=0.8214294752549057 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --pi_consistency_rampup_length=14 --pi_consistency_type=mse --pi_max_consistency_scaling=142.03110767666544 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=128 --use_xavier_init=True --warmup_epochs=22 --weight_decay=1e-07