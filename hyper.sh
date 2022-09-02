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
#dnn=vgg19
#dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &

# (1) learning rate
#lr=0.001
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#lr=0.01
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=4 ./train_cifar10.sh &
#lr=0.1
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=6 ./train_cifar10.sh &
#lr=1.0
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=8 ./train_cifar10.sh

# (2) batch size
#bs=512
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#bs=256
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=4 ./train_cifar10.sh &
#bs=128
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=6 ./train_cifar10.sh &
#bs=64
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=8 ./train_cifar10.sh

# (3) damping
#damping=0.003
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#damping=0.03
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=4 ./train_cifar10.sh &
#damping=0.3
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=6 ./train_cifar10.sh &
#damping=3.0
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=8 ./train_cifar10.sh

# (4) stat decay
#stat_decay=0.05
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=2 ./train_cifar10.sh &
#stat_decay=0.35
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=4 ./train_cifar10.sh &
#stat_decay=0.65
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=6 ./train_cifar10.sh &
#stat_decay=0.95
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 horovod=0 node_rank=8 ./train_cifar10.sh


# (5) ablation
#dnn=resnet110
#dataset=cifar10
#dnn=vgg19
#dataset=cifar100
#epochs=100 dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=-1 warmup_epochs=5 lr_schedule=step momentum=0.9 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=adasgd kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 horovod=0 node_rank=4 ./train_cifar10.sh &

# (6) autoencoder on mnist
#opt_name=sgd epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.6 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=1 ./train_mnist.sh
#opt_name=adamw weight_decay=0.05 epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.005 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=8 ./train_mnist.sh
#opt_name=sgd epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.06 kfac_name=kfac kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=1 ./train_mnist.sh
#opt_name=sgd epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.06 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=1 ./train_mnist.sh
#opt_name=shampoo epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.06 kfac_name=eva kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=8 ./train_mnist.sh
#opt_name=adagrad epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.002 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=8 ./train_mnist.sh

