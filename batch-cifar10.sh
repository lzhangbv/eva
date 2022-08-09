# Experiments on Cifar-10 and Cifar-100

dnn=vgg19
lr_schedule=step
stat_decay=0.95
kl_clip=0.001
bs=128
lr=0.1

epochs=50
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=2 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=3 ./train_cifar10.sh &


dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &


dnn=resnet110
epochs=50
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


epochs=100
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=0 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=1 autoaugment=1 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &


# Fine-tuning

dnn=efficientnet-b0
weight_decay=0.00005
lr_schedule=cosine
lr_decay="${lr_decay:-100}"
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
weight_decay=0
dataset=cifar10
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=7 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=8 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=9 ./train_cifar10.sh &

dataset=cifar100
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 fac=0 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=10 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=11 ./train_cifar10.sh &
#epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr weight_decay=$weight_decay use_pretrained_model=1 warmup_epochs=0 lr_decay=$lr_decay nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=$damping stat_decay=$stat_decay kl_clip=$kl_clip warmup_epochs=0 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=12 ./train_cifar10.sh &


