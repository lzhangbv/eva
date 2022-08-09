#!/bin/bash

# model training settings
dataset="${dataset:-cifar10}"
dnn="${dnn:-resnet110}"

# first-order hyper
batch_size="${batch_size:-128}"
base_lr="${base_lr:-0.1}"
lr_schedule="${lr_schedule:-step}"
lr_decay="${lr_decay:-0.35 0.75 0.9}"

epochs="${epochs:-100}"
warmup_epochs="${warmup_epochs:-5}"

momentum="${momentum:-0.9}"
opt_name="${opt_name:-sgd}"
weight_decay="${weight_decay:-0.0005}"

# tricks
label_smoothing="${label_smoothing:-0}"
mixup="${mixup:-0}"
cutmix="${cutmix:-0}"
cutout="${cutout:-0}"
autoaugment="${autoaugment:-0}"
use_pretrained_model="${use_pretrained_model:-0}"

# second-order hyper
kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-eva}"
damping="${damping:-0.001}"
stat_decay="${stat_decay:-0.95}"
kl_clip="${kl_clip:-0.001}"

horovod="${horovod:-1}"
params="--horovod $horovod --dataset $dataset --dir /datasets/cifar10 --model $dnn --batch-size $batch_size --base-lr $base_lr --lr-schedule $lr_schedule --lr-decay $lr_decay --epochs $epochs --warmup-epochs $warmup_epochs --momentum $momentum --opt-name $opt_name --weight-decay $weight_decay --label-smoothing $label_smoothing --mixup $mixup --cutmix $cutmix --autoaugment $autoaugment --cutout $cutout --use-pretrained-model $use_pretrained_model --kfac-update-freq $kfac --kfac-cov-update-freq $fac --kfac-name $kfac_name --stat-decay $stat_decay --damping $damping --kl-clip $kl_clip"

nworkers="${nworkers:-4}"
rdma="${rdma:-1}"
clusterprefix="${clusterprefix:-cluster}"

ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

script=examples/pytorch_cifar10_resnet.py

if [ "$horovod" = "1" ]; then
clusterprefix=$clusterprefix nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
