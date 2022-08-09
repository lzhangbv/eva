# shampoo

dnn=vgg19
lr_schedule=step
stat_decay=0.95
bs=128
lr=0.1

epochs=50
dataset=cifar10
opt_name=shampoo epochs=$epochs dnn=$dnn dataset=$dataset batch_size=$bs base_lr=$lr nworkers=4 kfac=1 fac=1 warmup_epochs=5 lr_schedule=$lr_schedule momentum=0.9 label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 horovod=0 node_rank=1 ./train_cifar10.sh
