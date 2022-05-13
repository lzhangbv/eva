#!/bin/bash

# model training settings
epochs="${epochs:-100}"
batch_size="${batch_size:-128}"

use_adam="${use_adam:-0}"
lr_mul="${lr_mul:-0.5}"
warmup="${warmup:-4000}"

base_lr="${base_lr:-0.000001}"
lr_decay="${lr_decay:-400}"
warmup_epochs="${warmup_epochs:-0}"

scale_emb_or_prj="${scale_emb_or_prj:-emb}"
n_layers="${n_layers:-2}"

kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-eva}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.03}"
exclude_parts="${exclude_parts:-''}"

horovod="${horovod:-1}"
params="--horovod $horovod --epoch $epochs --batch-size $batch_size  --lr-mul $lr_mul --n-warmup-steps $warmup --base-lr $base_lr --warmup-epochs $warmup_epochs --lr-decay $lr_decay --kfac-update-freq $kfac --kfac-cov-update-freq $fac --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name --exclude-parts ${exclude_parts} --data-pkl /datasets/m30k_deen_shr.pkl --label-smoothing --proj-share-weight --scale-emb-or-prj $scale_emb_or_prj --n-layers $n_layers --use-adam $use_adam"

# multi-node multi-gpu settings
nworkers="${nworkers:-8}"
rdma="${rdma:-1}"

script=examples/pytorch_multi30k_transformer.py

nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
