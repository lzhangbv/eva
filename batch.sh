# Shell script to run experiments of convergence performance and training efficiency. 
# Note:
#     (1) Before run convergence experiments, make sure SPEED=False in example scripts;
#         or make sure SPEED=True in example scripts before run efficiency experiments. 
#     (2) Please fine-tune the hyper-paramters in train_xxx.sh scripts. 
#     (3) Please configure the host files and the cluster environments in configs folder. 

fac=1
kfac=1
kfac_name=eva

# Convergence performance
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_cifar10.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_cifar100.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_multi30k.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name bash train_squad.sh


# Training efficiency
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet110 batch_size=128 nworkers=4 bash train_cifar10.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=vgg16 batch_size=128 nworkers=4 bash train_cifar100.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=resnet50 batch_size=32 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=densenet201 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 dnn=inceptionv4 batch_size=16 nworkers=64 bash train_imagenet.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=5 n_layers=6 batch_size=128 nworkers=8 bash train_multi30k.sh
#kfac=$kfac fac=$fac kfac_name=$kfac_name epochs=1 batch_size=4 nworkers=8 bash train_squad.sh


# Tuning the hyper-parameters
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu1cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu2cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu3cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.05 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu4cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu5cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu6cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu7cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.003 stat_decay=0.95 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu8cluster ./train_cifar10.sh &

#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu9cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu10cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu11cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.05 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu12cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=1 clusterprefix=gpu13cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu14cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.005 warmup_epochs=1 clusterprefix=gpu15cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.005 warmup_epochs=5 clusterprefix=gpu16cluster ./train_cifar10.sh &


# The tuned hyper-paramters
#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 clusterprefix=gpu1cluster ./train_cifar10.sh &
#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 clusterprefix=gpu2cluster ./train_cifar10.sh &

#epochs=100 dnn=vgg19 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 clusterprefix=gpu4cluster ./train_cifar10.sh &
#epochs=100 dnn=vgg19 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=kfac kfac=10 fac=10 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 clusterprefix=gpu5cluster ./train_cifar10.sh &
#epochs=100 dnn=vgg19 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 warmup_epochs=5 clusterprefix=gpu6cluster ./train_cifar10.sh &

#lr_decay="${lr_decay:-35 65 80}"
#epochs=100 dnn=vgg16 dataset=cifar100 batch_size=128 base_lr=0.1 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu1cluster ./train_cifar100.sh &
#epochs=100 dnn=vgg16 dataset=cifar100 batch_size=128 base_lr=0.1 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 warmup_epochs=5 clusterprefix=gpu1cluster ./train_cifar100.sh &

#lr_decay="${lr_decay:-100 150}"
#epochs=200 dnn=vgg16 dataset=cifar100 batch_size=128 base_lr=0.1 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu10cluster ./train_cifar100.sh &
#epochs=200 dnn=vgg16 dataset=cifar100 batch_size=128 base_lr=0.1 lr_decay=$lr_decay nworkers=4 kfac_name=eva kfac=0 warmup_epochs=5 clusterprefix=gpu10cluster ./train_cifar100.sh &

#epochs=100 n_layers=2 batch_size=128 base_lr=0.000001 nworkers=8 kfac_name=eva kfac=1 fac=1 damping=0.03 clusterprefix=gpu3cluster bash train_multi30k.sh 
#epochs=100 n_layers=2 batch_size=128 use_adam=1 lr_mul=0.5 warmup=4000 nworkers=8 kfac_name=eva kfac=0 clusterprefix=gpu3cluster bash train_multi30k.sh
#epochs=100 n_layers=6 batch_size=128 base_lr=0.000001 nworkers=8 kfac_name=eva kfac=1 fac=1 damping=0.03 clusterprefix=gpu3cluster bash train_multi30k.sh 
#epochs=100 n_layers=6 batch_size=128 use_adam=0 base_lr=0.000001 nworkers=8 kfac_name=eva kfac=0 clusterprefix=gpu9cluster bash train_multi30k.sh 

#epochs=3 model_type=bert batch_size=4 base_lr=0.000005 nworkers=8 kfac_name=eva kfac=1 fac=1 damping=0.03 clusterprefix=gpu3cluster bash train_squad.sh
#epochs=3 model_type=bert batch_size=4 use_adamw=1 base_lr=0.000005 nworkers=8 kfac_name=eva kfac=0 clusterprefix=gpu3cluster bash train_squad.sh

# On one worker
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu1cluster ./train_cifar10.sh &
#epochs=100 dnn=resnet110 dataset=cifar10 batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=0 warmup_epochs=5 clusterprefix=gpu2cluster ./train_cifar10.sh &

#epochs=100 dnn=vgg16 dataset=cifar100 batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.001 warmup_epochs=5 clusterprefix=gpu3cluster ./train_cifar100.sh &
#epochs=100 dnn=vgg16 dataset=cifar100 batch_size=512 base_lr=0.4 nworkers=1 kfac_name=eva kfac=0 warmup_epochs=5 clusterprefix=gpu2cluster ./train_cifar100.sh &

