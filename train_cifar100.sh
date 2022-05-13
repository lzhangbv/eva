#!/bin/bash

# model training settings
dataset=cifar100
dnn="${dnn:-vgg16}"
batch_size="${batch_size:-128}"
base_lr="${base_lr:-0.1}"
epochs="${epochs:-100}"
warmup_epochs="${warmup_epochs:-5}"

#lr_decay="${lr_decay:-80 120}"
#lr_decay="${lr_decay:-35 65 80 90}"
lr_decay="${lr_decay:-35 75 90}"

kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-eva}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
damping="${damping:-0.03}"

horovod="${horovod:-1}"
params="--horovod $horovod --dataset $dataset --dir /datasets/cifar10 --model $dnn --batch-size $batch_size --base-lr $base_lr --epochs $epochs --warmup-epochs $warmup_epochs --kfac-update-freq $kfac --kfac-cov-update-freq $fac --lr-decay $lr_decay --stat-decay $stat_decay --damping $damping --kfac-name $kfac_name --exclude-parts ${exclude_parts}"

nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

script=examples/pytorch_cifar10_resnet.py
nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
