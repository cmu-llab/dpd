# Semisupervised Neural Proto-Language Reconstruction

This repository accompanies the paper "Semisupervised Neural Proto-Language Reconstruction" at ACL 2024.

![dpd](https://github.com/cmu-llab/dpd/assets/38693485/bfe5967d-03d4-423d-b4b8-d441c56f2d9f)

> **Abstract:** Existing work implementing comparative reconstruction of ancestral languages (proto-languages) has usually required full supervision. However, historical reconstruction models are only of practical value if they can be trained with a limited amount of labeled data. We propose a semisupervised historical reconstruction task in which the model is trained on only a small amount of labeled data (cognate sets with proto-forms) and a large amount of unlabeled data (cognate sets without proto-forms). We propose a neural architecture for comparative reconstruction (DPD-BiReconstructor) incorporating an essential insight from linguists' comparative method: that reconstructed words should not only be reconstructable from their daughter words, but also deterministically transformable back into their daughter words. We show that this architecture is able to leverage unlabeled cognate sets to outperform strong semisupervised baselines on this novel task.

> **TL;DR:** We introduce the novel task of semisupervised protoform reconstruction and propose a neural architecture informed by historical linguists' comparative method, which outperforms baseline methods in almost all situations.

🚧 This repository is under construction. We plan on making more checkpoints available.

# Set up

## Python Environment

```
conda create --name dpd python=3.10.13 --yes
conda activate dpd

pip install editdistance==0.6.2 einops==0.6.1 huggingface-hub==0.16.4 lightning-utilities==0.8.0 lingpy==2.6.9 lingrex==1.3.0 matplotlib==3.7.1 numpy==1.24.3 pandas==2.0.2 Pillow==9.4.0 pytorch-lightning==2.0.4 sacrebleu==2.3.1 seaborn==0.12.2 tabulate==0.9.0 tokenizers==0.13.3 toml==0.10.2 torch==2.0.1 torchaudio==2.0.2 torchmetrics==0.11.4 torchshow==0.5.0 torchvision==0.15.2 tqdm==4.65.0 transformers==4.31.0 wandb==0.15.3 scikit-learn==1.4.0 scipy==1.12.0 lingpy==2.6.9 lingrex==1.3.0 newick==1.9.0 python-dotenv pandasql==0.7.3
pip install panphon@git+https://github.com/dmort27/panphon.git@6acd3833743a49e63941a0b740ee69eae1dafc1c
```

## GPU

We recommend using cuda GPUs. The `--cpu` flag allows running the code (`exp.py`) on the CPU.

## WandB

The code relies on [WandB](https://wandb.ai/) for results logging and checkpointing. To set up WandB, modify the `.env` file with your WandB entity and project in the following format:

```txt
WANDB_ENTITY = "awandbentity"
WANDB_PROJECT = "awandbproject"
```

# Dataset

Rom-phon is not licensed for redistribution. Please contact Ciobanu and Dinu (2014) to obtain the full Romance dataset. WikiHan is licensed under cc0 and is located in `data` with the name `chinese_wikihan2022`.

# Running Experiments

`exp.py` is the main script to run experiments. 

See `.sh` files under the `shs` directory for commands. We ran all the `.sh` files 10 times. Commands to replicate a single experiment can be found with the `.sh` file for the corresponding dataset, labeling setting, and group.

For example, running a 20% labeled group 1 Rom-phon Trans-DPD-ΠM experiment corresponds to running this command:

```sh
PROPORTION_LABELLED="0.2"
DATASET_SEED="1893608599"
GROUP_NAME="group1"

python exp.py --logmodel --tags paper $GROUP_NAME --vram_thresh 2000 --architecture=Transformer --batch_size=256 --beta1=0.9 --beta2=0.999 --check_val_every_n_epoch=1 --cringe_alpha=0.3294570624337493 --cringe_k=2 --d2p_dropout_p=0.3452534566458349 --d2p_embedding_dim=384 --d2p_feedforward_dim=512 --d2p_inference_decode_max_length=30 --d2p_max_len=128 --d2p_nhead=8 --d2p_num_decoder_layers=2 --d2p_num_encoder_layers=2 --d2p_recon_loss_weight=1.0333179173348133 --d2p_use_lang_separaters=True --dataset=Nromance_ipa --datasetseed=$DATASET_SEED --emb_pred_loss_weight=0.4612113930336585 --eps=1e-08 --lr=0.0006180685060490792 --max_epochs=206 --min_daughters=1 --p2d_all_lang_summary_only=True --p2d_dropout_p=0.31684496334382184 --p2d_embedding_dim=256 --p2d_feedforward_dim=512 --p2d_inference_decode_max_length=30 --p2d_loss_on_gold_weight=0.5989036965133778 --p2d_loss_on_pred_weight=0.8703013320652477 --p2d_max_len=128 --p2d_nhead=8 --p2d_num_decoder_layers=2 --p2d_num_encoder_layers=2 --pi_consistency_rampup_length=23 --pi_consistency_type=mse --pi_max_consistency_scaling=301.2992611249976 --proportion_labelled=$PROPORTION_LABELLED --skip_daughter_tone=False --skip_protoform_tone=False --strat=pimodel_bpall_cringe --test_val_batch_size=256 --universal_embedding=True --universal_embedding_dim=384 --use_xavier_init=True --warmup_epochs=29 --weight_decay=1e-07
```

# Notes

## Naming

- 100% supervised experiments are identified by `exclude_unlabelled` but are equivalent when unlabelled data is included.
- The WikiHan dataset could have been referred to as any of `wikihan`, `chinese_wikihan`, or `chinese_wikihan2022` in the code.
- The Rom-phon dataset could have been referred to as any of `Nromance_ipa`, `Nrom_ipa`, or `Nrom` in the code. The prefix `N` has no meaning.
- The strategy-architecture pairs have the following identifiers:
    - `GRUSupv` = GRU-SUPV
    - `GRUPi` = GRU-ΠM
    - `GRUBpall` = GRU-DPD
    - `GRUBpallPi` = GRU-DPD-ΠM
    - `TransSupv` = Trans-SUPV
    - `TransPi` = Trans-ΠM
    - `TransBpall` = Trans-DPD
    - `TransBpallPi` = Trans-DPD-ΠM
    - `GRUSupvBst` = GRU-SUPV-BST
    - `GRUPiBst` = GRU-ΠM-BST
    - `GRUBpallBst` = GRU-DPD-BST
    - `GRUBpallPiBst` = GRU-DPD-ΠM-BST
    - `TransSupvBst` = Trans-SUPV-BST
    - `TransPiBst` = Trans-ΠM-BST
    - `TransBpallBst` = Trans-DPD-BST
    - `TransBpallPiBst` = Trans-DPD-ΠM-BST
- Strategies can also have the following identifiers:
    - `supervised_only` = SUPV
    - `pimodel` = ΠM
    - `bpall_cringe` = DPD
    - `pimodel_bpall_cringe` = DPD-ΠM

## Implementation

- Transformer and GRU implementation are based on Kim et al. (2023) and Chang et al. (2022)’s PyTorch reimplementation of Meloni et al. (2021).
- These code segments are not used and can be safely ignored, including:
    - VAE
    - Beam search decode, which is not supported for DPD
    - `alignment_convolution_masking` and `convolution_masking_residue` when computing CRINGE loss
    - `GatedMLP`
    - `p2d_lang_embedding_when_decoder`, `p2d_prompt_mlp_with_one_hot_lang`, and `p2d_gated_mlp_by_target_lang` are never enabled
