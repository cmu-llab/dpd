# 🚧 under construction

The code is a bit outdated and doesn't yet include the additional experiments in the appendix. We'll update that soon.

We'll also be releasing checkpoints.

# Set up

## Python environment

```txt
editdistance==0.6.2
einops==0.6.1
huggingface-hub==0.16.4
lightning-utilities==0.8.0
lingpy==2.6.9
lingrex==1.3.0
matplotlib==3.7.1
numpy==1.24.3
pandas==2.0.2
panphon @ git+https://github.com/dmort27/panphon.git@6acd3833743a49e63941a0b740ee69eae1dafc1c
Pillow==9.4.0
pytorch-lightning==2.0.4
sacrebleu==2.3.1
seaborn==0.12.2
tabulate==0.9.0
tokenizers==0.13.3
toml==0.10.2
torch==2.0.1
torchaudio==2.0.2
torchmetrics==0.11.4
torchshow==0.5.0
torchvision==0.15.2
tqdm==4.65.0
transformers==4.31.0
wandb==0.15.3
scikit-learn==1.4.0
scipy==1.12.0
lingpy==2.6.9
lingrex==1.3.0
newick==1.9.0
```

## GPU

We recommend using cuda GPUs. The `--cpu` flag allows running the code (`exp.py`) on the CPU.

## WandB

The code rely on WandB for results logging and checkpointing. To set up WandB, please create a `.env` file with the following information

```txt
WANDB_ENTITY = "your wandb entity"
WANDB_PROJECT = "a project on wandb"
```

# Dataset

Rom-phon is not licenced for redistribution. Please contact Ciobanu and Dinu (2014) to obtain the full Romance dataset. WikiHan is licenced under cc0 and is located in `data` with the name `chinese_wikihan2022`.

# Running experiments

See `.sh` files under the `shs` directory. Our data is collected from running all the `.sh` files 10 times. Commands to replicate a single experiment can be found with the `.sh` file for the corresponding dataset, label setting, and group. For WikiHan, Bootstrapping and non-Bootstrapping experiments are split into two scripts, identified by whether `_bst_` is present in the name.

For example, running a 20% labeled group 1 Rom-phon Trans-DPD-Pi experiment corresponds to running this command:

```sh
PROPORTION_LABELLED="0.2"
DATASET_SEED="1893608599"
GROUP_NAME="group1"

python exp.py --logmodel --tags paper $GROUP_NAME --vram_thresh 2000 --architecture=Transformer --batch_size=256 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.3294570624337493 --cringe_k=2 --d2p_dropout_p=0.3452534566458349 --d2p_embedding_dim=384 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=30 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=1.0333179173348133 --d2p_use_lang_separaters=True --dataset=Nromance_ipa --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.4612113930336585 --eps=1e-08 --lr=0.0006180685060490792 --max_epochs=206 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.31684496334382184 --p2d_embedding_dim=256 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=30 --p2d_loss_on_gold_weight=0.5989036965133778 --p2d_loss_on_pred_weight=0.8703013320652477 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --pi_consistency_rampup_length=23 --pi_consistency_type=mse --pi_max_consistency_scaling=301.2992611249976 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=384 --use_xavier_init=True --warmup_epochs=29 --weight_decay=1e-07
```

# Notes

- Transformer and GRU implementation are based on Kim et al. (2023) and Chang et al. (2022)’s PyTorch reimplementation of Meloni et al. (2021), both of which are intended for research purpose. Our code is indented for the purpose of replication and research.
- These code segments are not used for this paper and can be safely ignored, including:
    - VAE
    - Beam search decode, which is not supported for DPD
    - `alignment_convolution_masking` and `convolution_masking_residue` when computing CRINGE loss
    - `GatedMLP`
    - `p2d_lang_embedding_when_decoder`, `p2d_prompt_mlp_with_one_hot_lang`, and `p2d_gated_mlp_by_target_lang` are never enabled
