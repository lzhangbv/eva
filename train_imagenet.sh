#!/bin/bash

# model training settings
dnn="${dnn:-resnet50}"
batch_size="${batch_size:-32}"
base_lr="${base_lr:-0.0125}" 
epochs="${epochs:-55}"
warmup_epochs="${warmup_epochs:-5}"
opt_name="${opt_name:-sgd}"

if [ "$epochs" = "90" ]; then
lr_decay="${lr_decay:-30 60 80}"
else
lr_decay="${lr_decay:-25 35 40 45 50}"
fi

kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-eva}"
exclude_parts="${exclude_parts:-''}"
stat_decay="${stat_decay:-0.95}"
kl_clip="${kl_clip:-0.001}"
damping="${damping:-0.03}"
ngrads="${ngrads:-32}"

horovod="${horovod:-1}"
params="--horovod $horovod --model $dnn --base-lr $base_lr --epochs $epochs --lr-decay $lr_decay --kfac-update-freq $kfac --kfac-cov-update-freq $fac --kfac-name $kfac_name --stat-decay $stat_decay --damping $damping --kl-clip $kl_clip --exclude-parts ${exclude_parts} --batch-size $batch_size --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val --opt-name $opt_name --ngrads $ngrads"

# multi-node multi-gpu settings
nworkers="${nworkers:-64}"
rdma="${rdma:-1}"
clusterprefix="${clusterprefix:-cluster}"

ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

script=examples/pytorch_imagenet_resnet.py

if [ "$horovod" = "1" ]; then
clusterprefix=$clusterprefix nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
