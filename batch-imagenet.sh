# Experiments on ImageNet-1k

dnn=resnet50
bs=64
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh

dnn=resnet152
bs=16
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh

dnn=densenet121
bs=32
#dnn=densenet201
#bs=16
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh

dnn=inceptionv4
bs=32
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh

dnn=vit-b16
bs=16
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=16 bash train_imagenet.sh

# efficiency on multiple GPUe
dnn=resnet50
#opt_name=sgd fac=0 kfac=0 epochs=1 dnn=$dnn batch_size=96 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=1 kfac_name=kfac dnn=$dnn batch_size=64 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=sgd fac=1 kfac=1 epochs=1 kfac_name=eva dnn=$dnn batch_size=96 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=1 dnn=$dnn batch_size=64 nworkers=32 horovod=0 rdma=0 ngpu_per_node=4 node_count=8 node_rank=9 bash train_imagenet.sh

# convergence
dnn=resnet50
#opt_name=sgd fac=1 kfac=1 epochs=55 kfac_name=eva damping=0.001 dnn=$dnn base_lr=0.05 batch_size=96 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=sgd fac=50 kfac=50 epochs=55 kfac_name=kfac damping=0.001 dnn=$dnn base_lr=0.05 batch_size=64 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=sgd fac=0 kfac=0 epochs=100 dnn=$dnn base_lr=0.05 batch_size=96 nworkers=32 horovod=1 rdma=0 bash train_imagenet.sh
#opt_name=shampoo fac=50 kfac=50 epochs=60 dnn=$dnn base_lr=0.05 batch_size=64 nworkers=32 horovod=0 rdma=0 ngpu_per_node=4 node_count=8 node_rank=1 bash train_imagenet.sh &

# ablation
#opt_name=sgd fac=1 kfac=1 epochs=55 kfac_name=eva damping=0.001 momentum=0.0 dnn=$dnn base_lr=0.05 batch_size=96 nworkers=32 horovod=0 rdma=0 ngpu_per_node=4 node_count=8 node_rank=1 bash train_imagenet.sh &
#opt_name=sgd fac=1 kfac=1 epochs=55 kfac_name=adasgd damping=0.001 dnn=$dnn base_lr=0.05 batch_size=96 nworkers=32 horovod=0 rdma=0 ngpu_per_node=4 node_count=8 node_rank=9 bash train_imagenet.sh

