#!/bin/bash

# usage: nworkers=$nworkers rdma=$rdma script=$script params=$params bash launch_horovod.sh

# config and python script
source configs/envs.conf
script="${script:-}"
params="${params:-}"

# multi-node multi-gpu setting
clusterprefix="${clusterprefix:-cluster}"
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

if [ "$rdma" = "0" ]; then
net_params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0"
else
net_params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    -x HOROVOD_CACHE_CAPACITY=0"
fi

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile configs/$clusterprefix${nworkers} -bind-to none -map-by slot $net_params $PY $script $params
