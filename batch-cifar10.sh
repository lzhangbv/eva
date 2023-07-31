# Experiments on Cifar-10 and Cifar-100


# (1) semi-KFAC (FOOF) vs. semi-Eva (Eva-f) with KL-normalization
bs=128 #128, 256, 512
lr=0.1
epochs=100
stat_decay=0.95
kl_clip=1
lrs=cosine #step, cosine

#dnn=vgg19
#dataset=cifar100
dnn=resnet110
dataset=cifar10

# SGD
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &

# Eva-f
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &

# FOOF
#topk=0 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=13 ./train_cifar10.sh &
#topk=1 epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 kfac_name=semi-kfac kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=0.001 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &


# (2) Shampoo vs. vectorized Shampoo (Eva-s)

# Eva-s
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=vshampoo kfac=1 fac=1 damping=0.03 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &

# Shampoo
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=0.1 nworkers=4 opt_name=shampoo kfac=50 fac=50 damping=0.03 stat_decay=1 kl_clip=1 warmup_epochs=5 lr_schedule=$lrs momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &


# (3) Time and memory
dataset=cifar10
dnn=vgg19 #vgg19, resnet110
bs=512
#dnn=wrn28-10
#bs=256

#kfac=1 fac=1 kfac_name=semi-eva epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=2 ngpu_per_node=1 ./train_cifar10.sh
#kfac=1 fac=1 opt_name=vshampoo epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=2 ngpu_per_node=1 ./train_cifar10.sh


