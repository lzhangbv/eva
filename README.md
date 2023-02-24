# Eva: An Efficient Second-Order Algorithm with Sublinear Memory Cost

The Eva code was originally forked from Lin Zhang's [kfac-pytorch](https://github.com/lzhangbv/kfac_pytorch). 

## Install

### Requirements

PyTorch and Horovod are required to use K-FAC.

This code is validated to run with PyTorch-1.10.0, Horovod-0.21.0, CUDA-10.2, cuDNN-7.6, and NCCL-2.6.4. 

### Installation

```
$ git clone https://github.com/lzhangbv/eva.git
$ cd eva
$ pip install -r requirements.txt
$ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
```

If pip installation failed, please try to upgrade pip via `pip install --upgrade pip`. If Horovod installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html). 

## Usage

The Distributed Eva can be easily added to exisiting training scripts that use PyTorch's Distributed Data Parallelism.

```Python
from kfac import Eva
... 
model = torch.nn.parallel.DistributedDataParallel(...)
optimizer = optim.SGD(model.parameters(), ...)
preconditioner = Eva(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    preconditioner.step()
    optimizer.step()
...
```

Note that the Eva performs a preconditioning step before model update. 

For the convenience of experiments, we support choosing Eva and K-FAC using kfac.get_kfac_module. 

```Python
import kfac
...
KFAC = kfac.get_kfac_module(kfac='kfac')
preconditioner = KFAC(model, ...)
...
```

Note that the 'kfac' represent original K-FAC algorithms proposed in [ICML 2015](https://arxiv.org/abs/1503.05671) and [ICML 2016](https://arxiv.org/abs/1602.01407) with some modifications to improve efficiency. 

## Configure the cluster settings

Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster enviroments


## Run experiments

```
$ mkdir logs
$ bash batch-cifar10.sh
```

See `python examples/pytorch_{dataset}_{model}.py --help` for a full list of hyper-parameters.
Note: if `--kfac-update-freq 0`, the K-FAC Preconditioning is skipped entirely, i.e. training is just with SGD or Adam. 

Make sure the datasets were prepared in correct dirs (e.g., /datasets/cifar10) before running the experiments. We downloaded Cifar-10, Cifar-100, and Imagenet datasets via Torchvision's [Datasets](https://pytorch.org/vision/stable/datasets.html). 

## Citation

```
@inproceedings{zhang2023eva,
  title={Eva: Practical Second-order Optimization with Kronecker-vectorized Approximation},
  author={Zhang, Lin and Shi, Shaohuai and Li, Bo},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
