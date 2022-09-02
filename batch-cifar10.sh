# Experiments on Cifar-10 and Cifar-100


# (1) SGD, K-FAC, and Eva
lr_schedule=step
stat_decay=0.95
kl_clip=0.001
bs=128
lr=0.1

epochs=50
dnn=vgg19
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &

epochs=50
dnn=resnet110
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=16 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=14 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &

epochs=50
dnn=wrn28-10
dataset=cifar10
lr_schedule=step
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &


# (2) Fine-tuning
dnn=efficientnet-b0
weight_decay=0.00005
lr_schedule=cosine
stat_decay=0.95
kl_clip=0
damping=0.03
bs=24
lr=0.01

epochs=20
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &

dnn=vit-b16
lr=0.001
weight_decay=0
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &


# (3) Shampoo, Adagrad, AdamW, M-FAC
bs=128
epochs=100

# Shampoo
lr=0.1
lr_schedule=step
#opt_name=shampoo epochs=$epochs dnn=vgg19 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=10 fac=10 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &
#opt_name=shampoo epochs=$epochs dnn=resnet110 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=10 fac=10 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#opt_name=shampoo epochs=$epochs dnn=wrn28-10 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=10 fac=10 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &

# Adagrad
lr=0.01
lr_schedule=cosine
#opt_name=adagrad epochs=$epochs dnn=vgg19 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=4 ./train_cifar10.sh &
#opt_name=adagrad epochs=$epochs dnn=resnet110 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ./train_cifar10.sh &
#opt_name=adagrad epochs=$epochs dnn=wrn28-10 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=6 ./train_cifar10.sh &

# AdamW
lr=0.001
weight_decay=0.05
lr_schedule=cosine
#opt_name=adamw epochs=$epochs dnn=vgg19 dataset=cifar10 batch_size=$bs base_lr=$lr weight_decay=$weight_decay nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#opt_name=adamw epochs=$epochs dnn=resnet110 dataset=cifar10 batch_size=$bs base_lr=$lr weight_decay=$weight_decay nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#opt_name=adamw epochs=$epochs dnn=wrn28-10 dataset=cifar10 batch_size=$bs base_lr=$lr weight_decay=$weight_decay nworkers=4 kfac=0 fac=0 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &

# M-FAC
lr=0.06
lr_schedule=cosine
#opt_name=mfac epochs=$epochs dnn=vgg19 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=1 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ngpu_per_node=1 ./train_cifar10.sh &
#opt_name=mfac epochs=$epochs dnn=resnet110 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=1 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ngpu_per_node=1 ./train_cifar10.sh &
#opt_name=mfac epochs=$epochs dnn=wrn28-10 dataset=cifar10 batch_size=$bs base_lr=$lr nworkers=1 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ngpu_per_node=1 ./train_cifar10.sh &


# (4) efficiency on one GPU
dataset=cifar10
dnn=resnet110 #vgg19, resnet110
bs=512
#dnn=wrn28-10
#bs=256

#kfac=0 epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh
#kfac=1 fac=1 kfac_name=kfac epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh
#kfac=10 fac=10 kfac_name=kfac epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh
#kfac=1 fac=1 opt_name=shampoo epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh
#kfac=10 fac=10 opt_name=shampoo epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh
#kfac=1 fac=1 kfac_name=eva epochs=1 dnn=$dnn dataset=$dataset batch_size=$bs horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh


# (5) time to convergence on one GPU
dnn=vgg19 #vgg19, resnet110
dataset=cifar10
#epochs=200 dnn=$dnn dataset=$dataset batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=6 ngpu_per_node=1 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ngpu_per_node=1 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=512 base_lr=0.4 nworkers=1 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh &
#opt_name=shampoo epochs=100 dnn=$dnn dataset=$dataset batch_size=512 base_lr=0.4 nworkers=1 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=5 ngpu_per_node=1 ./train_cifar10.sh &

dnn=wrn28-10
#epochs=200 dnn=$dnn dataset=$dataset batch_size=256 base_lr=0.4 nworkers=1 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=6 ngpu_per_node=1 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=256 base_lr=0.4 nworkers=1 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=7 ngpu_per_node=1 ./train_cifar10.sh &
#epochs=100 dnn=$dnn dataset=$dataset batch_size=256 base_lr=0.4 nworkers=1 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=8 ngpu_per_node=1 ./train_cifar10.sh &
#opt_name=shampoo epochs=100 dnn=$dnn dataset=$dataset batch_size=256 base_lr=0.4 nworkers=1 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 lr_schedule=step momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=7 ngpu_per_node=1 ./train_cifar10.sh &

