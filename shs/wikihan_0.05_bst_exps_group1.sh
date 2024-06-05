PROPORTION_LABELLED="0.05"
DATASET_SEED="2706283079"
GROUP_NAME="group1"

# wikihan_GRU-SUPV-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=GRU --batch_size=128 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.0069000668307560645 --bootstrapping_min_epoch=38 --bootstrapping_pseudolabelling_cap=75 --check_val_every_n_epoch=1 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.2335640765090932 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=128 --d2p_num_encoder_layers=2 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0008284986520540648 --max_epochs=238 --min_daughters=1 --p2d_all_lang_summary_only=True --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=supervised_only --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=22 --weight_decay=0

# wikihan_GRU-Pi-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=GRU --batch_size=64 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.008731231971396676 --bootstrapping_min_epoch=7 --bootstrapping_pseudolabelling_cap=87 --check_val_every_n_epoch=1 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.313734950931792 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=128 --d2p_num_encoder_layers=2 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0005646293769171116 --max_epochs=253 --min_daughters=1 --p2d_all_lang_summary_only=True --pi_consistency_rampup_length=11 --pi_consistency_type=mse --pi_max_consistency_scaling=222.09001902562 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=16 --weight_decay=0

# wikihan_GRU-DPD-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=GRU --batch_size=64 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.007430832718872335 --bootstrapping_min_epoch=29 --bootstrapping_pseudolabelling_cap=41 --check_val_every_n_epoch=1 --cringe_alpha=0.5720240351003496 --cringe_k=1 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.1998523734296692 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=64 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.769611124922535 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.6173156812104053 --eps=1e-08 --lr=0.0008380400425674595 --max_epochs=384 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_decode_mode=greedy_search --p2d_dropout_p=0.3144957314881693 --p2d_embedding_dim=384 --p2d_feedforward_dim=512 --p2d_gated_mlp_by_target_lang=False --p2d_inference_decode_max_length=15 --p2d_lang_embedding_when_decoder=False --p2d_loss_on_gold_weight=0.09114688022442874 --p2d_loss_on_pred_weight=0.9654484219687338 --p2d_model_size=128 --p2d_num_encoder_layers=2 --p2d_prompt_mlp_with_one_hot_lang=False --p2d_use_bidirectional_encoder=False --p2d_use_vae_latent=False --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=128 --use_xavier_init=True --warmup_epochs=2 --weight_decay=0

# wikihan_GRU-DPD-Pi-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=GRU --batch_size=64 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.005347801618967403 --bootstrapping_min_epoch=24 --bootstrapping_pseudolabelling_cap=49 --check_val_every_n_epoch=1 --cringe_alpha=0.3731816252375824 --cringe_k=3 --d2p_decode_mode=greedy_search --d2p_dropout_p=0.24713920042897308 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_lang_embedding_when_decoder=True --d2p_model_size=64 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.5395786105243536 --d2p_use_bidirectional_encoder=False --d2p_use_lang_separaters=True --d2p_use_vae_latent=False --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.3858486594098394 --eps=1e-08 --lr=0.0008118445065167122 --max_epochs=253 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_decode_mode=greedy_search --p2d_dropout_p=0.15461708923273426 --p2d_embedding_dim=128 --p2d_feedforward_dim=512 --p2d_gated_mlp_by_target_lang=False --p2d_inference_decode_max_length=15 --p2d_lang_embedding_when_decoder=False --p2d_loss_on_gold_weight=0.6411328114728092 --p2d_loss_on_pred_weight=1.4675145638252989 --p2d_model_size=128 --p2d_num_encoder_layers=2 --p2d_prompt_mlp_with_one_hot_lang=False --p2d_use_bidirectional_encoder=False --p2d_use_vae_latent=False --pi_consistency_rampup_length=23 --pi_consistency_type=mse --pi_max_consistency_scaling=181.6923169339665 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=256 --use_xavier_init=True --warmup_epochs=27 --weight_decay=0

# wikihan_Trans-SUPV-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=Transformer --batch_size=256 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.002606939128705346 --bootstrapping_min_epoch=16 --bootstrapping_pseudolabelling_cap=54 --check_val_every_n_epoch=1 --d2p_dropout_p=0.19492134882696244 --d2p_embedding_dim=256 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0009074301806048172 --max_epochs=374 --min_daughters=1 --p2d_all_lang_summary_only=True --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=supervised_only --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=17 --weight_decay=1e-07

# wikihan_Trans-Pi-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=Transformer --batch_size=256 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.009312482077586606 --bootstrapping_min_epoch=4 --bootstrapping_pseudolabelling_cap=79 --check_val_every_n_epoch=1 --d2p_dropout_p=0.18770754480615 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --eps=1e-08 --lr=0.0008765249546763417 --max_epochs=341 --min_daughters=1 --p2d_all_lang_summary_only=True --pi_consistency_rampup_length=8 --pi_consistency_type=mse --pi_max_consistency_scaling=184.50859345367095 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel --test_val_batch_size=256 --use_xavier_init=True --warmup_epochs=26 --weight_decay=1e-07

# wikihan_Trans-DPD-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=Transformer --batch_size=64 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.003974880336136077 --bootstrapping_min_epoch=25 --bootstrapping_pseudolabelling_cap=86 --check_val_every_n_epoch=1 --cringe_alpha=0.6781894994402957 --cringe_k=3 --d2p_dropout_p=0.15742691094738195 --d2p_embedding_dim=384 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=1.3381186249131989 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.12957822944766734 --eps=1e-08 --lr=0.000664206321867968 --max_epochs=390 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.17517469748885092 --p2d_embedding_dim=256 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=15 --p2d_loss_on_gold_weight=0.4811087594748163 --p2d_loss_on_pred_weight=0.8611519773739926 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=256 --use_xavier_init=True --warmup_epochs=31 --weight_decay=0

# wikihan_Trans-DPD-Pi-BST
python exp.py --logmodel --tags paper $GROUP_NAME --bootstrapping --vram_thresh 2000 --architecture=Transformer --batch_size=64 --beta1=0.9 --beta2=0.999 --bootstrapping_alpha=0.7 --bootstrapping_inference_batch_size=256 --bootstrapping_log_prob_thresh=-0.007041616859797583 --bootstrapping_min_epoch=11 --bootstrapping_pseudolabelling_cap=35 --check_val_every_n_epoch=1 --cringe_alpha=0.30564947115065355 --cringe_k=3 --d2p_dropout_p=0.16923789976947598 --d2p_embedding_dim=128 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=15 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=0.6952336133258257 --d2p_use_lang_separaters=True --dataset=chinese_wikihan2022 --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.21819678779026333 --eps=1e-08 --lr=0.0008475335815977582 --max_epochs=259 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.3339079271308638 --p2d_embedding_dim=384 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=15 --p2d_loss_on_gold_weight=0.2464499345446821 --p2d_loss_on_pred_weight=1.3218183843125768 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --pi_consistency_rampup_length=27 --pi_consistency_type=mse --pi_max_consistency_scaling=88.69937192237501 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=256 --use_xavier_init=True --warmup_epochs=22 --weight_decay=0