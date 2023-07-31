# Experiments on ImageNet-1k

dnn=resnet50
bs=64

dnn=inceptionv4
bs=32

dnn=vit-b16
bs=16

#opt_name=sgd kfac_name=semi-eva fac=1 kfac=1 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=2 bash train_imagenet.sh
#opt_name=vshampoo fac=1 kfac=1 epochs=1 dnn=$dnn batch_size=$bs nworkers=1 horovod=0 ngpu_per_node=1 node_count=1 node_rank=2 bash train_imagenet.sh

