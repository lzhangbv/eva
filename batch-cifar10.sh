# Experiments on Cifar-10 and Cifar-100


# (1) SGD vs. Eva (semi-Eva) with KL-normalization
bs=128 #128, 256, 512
lr=0.1
epochs=100
stat_decay=0.95
kl_clip=1

dnn=vgg19
dataset=cifar10
#dnn=resnet110
#dataset=cifar100
#eva=eva
eva=semi-eva

#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=cosine momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=cosine momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=linear momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=linear momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=polynomial momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=polynomial momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=15 ./train_cifar10.sh &


# semi-KFAC
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=kfac kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=kfac kfac=50 fac=50 damping=0.03 stat_decay=$stat_decay kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &

#topk=0 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#topk=1 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#topk=16 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &
#topk=64 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=13 ./train_cifar10.sh &
#topk=128 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=14 ./train_cifar10.sh &
#topk=256 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=15 ./train_cifar10.sh &
#topk=512 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &


# (2) the effects of extremely small damping
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=1e-5 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=1e-8 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=$eva kfac=1 fac=1 damping=0 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh

#opt_name=sgd epochs=100 dnn=autoencoder dataset=mnist batch_size=1000 base_lr=0.06 kfac_name=$eva kfac=1 fac=1 damping=0 stat_decay=0.95 kl_clip=1 warmup_epochs=0 lr_schedule=linear momentum=0.9 horovod=0 node_rank=8 ./train_mnist.sh

# (3) Shampoo vs. vectorized Shampoo
bs=128 #128, 256, 512
lr=0.1
lrs=cosine
epochs=100
damping=0.03

dnn=vgg19
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=$damping stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=shampoo kfac=10 fac=10 damping=$damping stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=0.003 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=0.3 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=3 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &


dnn=resnet110
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=$damping stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=shampoo kfac=10 fac=10 damping=$damping stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=0.003 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=0.3 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=3 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &

