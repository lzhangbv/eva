#!/bin/bash

# model training settings
dataset="${dataset:-mnist}"
dnn="${dnn:-autoencoder}"
seed="${seed:-42}"

# first-order hyper
batch_size="${batch_size:-1000}"
base_lr="${base_lr:-0.1}"
lr_schedule="${lr_schedule:-linear}"
lr_decay="${lr_decay:-0.35 0.75 0.9}"

epochs="${epochs:-100}"
warmup_epochs="${warmup_epochs:-5}"

momentum="${momentum:-0.9}"
opt_name="${opt_name:-sgd}"
weight_decay="${weight_decay:-0.0005}"

# second-order hyper
kfac="${kfac:-1}"
fac="${fac:-1}"
kfac_name="${kfac_name:-eva}"
damping="${damping:-0.03}"
stat_decay="${stat_decay:-0.95}"
kl_clip="${kl_clip:-0.001}"

horovod="${horovod:-1}"
params="--horovod $horovod --dataset $dataset --dir /datasets --model $dnn --batch-size $batch_size --base-lr $base_lr --lr-schedule $lr_schedule --lr-decay $lr_decay --epochs $epochs --warmup-epochs $warmup_epochs --momentum $momentum --opt-name $opt_name --weight-decay $weight_decay --kfac-update-freq $kfac --kfac-cov-update-freq $fac --kfac-name $kfac_name --stat-decay $stat_decay --damping $damping --kl-clip $kl_clip --seed $seed"

nworkers="${nworkers:-1}"
rdma="${rdma:-1}"
clusterprefix="${clusterprefix:-cluster}"

ngpu_per_node="${ngpu_per_node:-1}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

script=examples/pytorch_mnist_autoencoder.py

if [ "$horovod" = "1" ]; then
clusterprefix=$clusterprefix nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh
else
ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank script=$script params=$params bash launch_torch.sh
fi
