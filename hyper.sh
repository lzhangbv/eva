# Hyper-parameter study

# default
epochs=100
lr_schedule=step
bs=128
lr=0.1
damping=0.03
stat_decay=0.95
kl_clip=0.001

dnn=resnet110
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &

# (1) learning rate
#lr=0.001
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=9 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=10 ./train_cifar10.sh &
#lr=0.01
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=12 ./train_cifar10.sh &
#lr=0.1
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=13 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=14 ./train_cifar10.sh &
#lr=1.0
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=15 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=16 ./train_cifar10.sh &

# (2) batch size
#bs=64
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=9 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=10 ./train_cifar10.sh &
#bs=128
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=12 ./train_cifar10.sh &
#bs=256
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=13 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=14 ./train_cifar10.sh &
#bs=512
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=15 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=16 ./train_cifar10.sh &

# (3) damping
damping=0.003
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=9 ./train_cifar10.sh &
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=10 ./train_cifar10.sh &
damping=0.03
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=11 ./train_cifar10.sh &
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=12 ./train_cifar10.sh &
damping=0.3
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=13 ./train_cifar10.sh &
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=14 ./train_cifar10.sh &
damping=3.0
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=15 ./train_cifar10.sh &
epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=16 ./train_cifar10.sh &

