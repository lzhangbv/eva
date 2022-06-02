
#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=step label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu1cluster ./train_cifar10.sh &
#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=step label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu2cluster ./train_cifar10.sh &

#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu3cluster ./train_cifar10.sh &
#epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=0 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu4cluster ./train_cifar10.sh

epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu1cluster ./train_cifar10.sh &
epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=0 autoaugment=0 cutout=0 clusterprefix=gpu2cluster ./train_cifar10.sh &


epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=0 cutout=0 clusterprefix=gpu3cluster ./train_cifar10.sh &
epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=0 cutout=0 clusterprefix=gpu4cluster ./train_cifar10.sh &

epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=1 cutout=0 clusterprefix=gpu5cluster ./train_cifar10.sh &
epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=1 cutout=0 clusterprefix=gpu6cluster ./train_cifar10.sh &

epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=1 cutout=1 clusterprefix=gpu7cluster ./train_cifar10.sh &
epochs=100 dnn=wrn28-10 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 lr_schedule=cosine label_smoothing=1 cutmix=1 autoaugment=1 cutout=1 clusterprefix=gpu8cluster ./train_cifar10.sh &

#epochs=100 dnn=nf-resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=0 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 use_polynomial_decay=1 label_smoothing=1 cutmix=1 autoaugment=1 clusterprefix=gpu1cluster ./train_cifar10.sh &
#epochs=100 dnn=nf-resnet110 dataset=cifar10 batch_size=128 base_lr=0.1 nworkers=4 kfac_name=eva kfac=1 fac=1 damping=0.03 stat_decay=0.95 kl_clip=0.01 warmup_epochs=5 use_polynomial_decay=1 label_smoothing=1 cutmix=1 autoaugment=1 clusterprefix=gpu2cluster ./train_cifar10.sh

