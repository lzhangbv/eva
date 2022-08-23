from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import wandb
wandb = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import torch.multiprocessing as mp

from utils import *

import kfac
import kfac.backend as backend #don't use a `from` import
from shampoo.shampoo import Shampoo
from mfac.optim import MFAC

import horovod.torch as hvd
import torch.distributed as dist

SPEED = False

def initialize():
    # Training Parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='autoencoder')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--horovod', type=int, default=1, metavar='N',
                        help='whether use horovod as communication backend (default: 1)')

    # SGD Parameters
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-schedule', type=str, default='step', 
                        choices=['step', 'polynomial', 'cosine'], help='learning rate schedules')
    parser.add_argument('--lr-decay', nargs='+', type=float, default=[0.5, 0.75],
                        help='epoch intervals to decay lr when using step schedule')
    parser.add_argument('--lr-decay-alpha', type=float, default=0.1,
                        help='learning rate decay alpha')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--label-smoothing', type=int, default=0, metavar='WE',
                        help='label smoothing (default: 0)')
    parser.add_argument('--opt-name', type=str, default='sgd', metavar='WE',
                        help='choose base optimizer [sgd, adam, shampoo, mfac]')

    # KFAC Parameters
    parser.add_argument('--kfac-type', type=str, default='Femp', 
                        help='choices: F1mc or Femp') 
    parser.add_argument('--kfac-name', type=str, default='inverse',
                        help='choices: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
                        help='choices: ComputeFactor,CommunicateFactor,ComputeInverse,CommunicateInverse')
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.03,
                        help='KFAC damping factor (defaultL 0.03)')
    parser.add_argument('--kl-clip', type=float, default=0.01,
                        help='KL clip (default: 0.01)')

    # Other Parameters
    parser.add_argument('--log-dir', default='./logs',
                        help='log directory')
    parser.add_argument('--dir', default='/datasets',
                        help='location of the training dataset in the local filesystem (will be downloaded if needed)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    
    # local_rank: (1) parse argument as folows in torch.distributed.launch; (2) read from environment in torch.distributed.run, i.e. local_rank=int(os.environ['LOCAL_RANK'])
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()


    # Training Settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_kfac = True if (args.kfac_update_freq > 0 and args.opt_name == 'sgd') else False
    
    if args.lr_decay[0] < 1.0: # epoch number percent
        args.lr_decay = [args.epochs * p for p in args.lr_decay]
    
    # Comm backend init
    # args.horovod = False
    if args.horovod:
        hvd.init()
        backend.init("Horovod")
    else:
        dist.init_process_group(backend='nccl', init_method='env://')
        backend.init("Torch")
    
    args.local_rank = backend.comm.local_rank()
    logger.info("GPU %s out of %s GPUs", backend.comm.rank(), backend.comm.size())

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # Logging Settings
    algo = args.kfac_name if args.use_kfac else args.opt_name
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir,
        '{}_{}_ep{}_bs{}_lr{}_gpu{}_kfac{}_{}.log'.format(args.dataset, args.model, args.epochs, args.batch_size, args.base_lr, backend.comm.size(), args.kfac_update_freq, algo))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    args.verbose = True if backend.comm.rank() == 0 else False
    
    if args.verbose:
        logger.info("torch version: %s", torch.__version__)
        logger.info(args)

    if args.verbose and wandb:
        wandb.init(project="kfac", entity="hkust-distributedml", name=logfile, config=args)
    
    return args


def get_dataset(args):
    # Load Cifar10
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST(root=args.dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=args.dir, train=False, download=False, transform=transform)

    # Use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    # train_loader = MultiEpochsDataLoader(train_dataset,
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # Use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    # test_loader = MultiEpochsDataLoader(test_dataset, 
    test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)
    
    return train_sampler, train_loader, test_sampler, test_loader


class Autoencoder(nn.Module):
    def __init__(self, mode):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 30)
        self.fc5 = nn.Linear(30, 250)
        self.fc6 = nn.Linear(250, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, 784)
   
    def forward(self, inputs):
        # encoder
        x = inputs.view(inputs.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # decoder
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x.view(-1, 28, 28)

def get_model(args):
    model = Autoencoder()

    if args.cuda:
        model.cuda()

    # Optimizer
    criterion = nn.MSELoss()

    args.base_lr = args.base_lr * backend.comm.size()
    if args.opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), 
                lr=args.base_lr, 
                betas=(0.9, 0.999), 
                weight_decay=args.weight_decay)
    elif args.opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), 
                lr=args.base_lr, 
                betas=(0.9, 0.999), 
                weight_decay=args.weight_decay)
    elif args.opt_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), 
                lr=args.base_lr, 
                weight_decay=args.weight_decay)
    elif args.opt_name == "shampoo":
        optimizer = Shampoo(params=model.parameters(), 
                    lr=args.base_lr, 
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay, 
                    statistics_compute_steps=args.kfac_cov_update_freq, 
                    preconditioning_compute_steps=args.kfac_update_freq)
    elif args.opt_name == 'mfac':
        assert backend.comm.size() == 1, "does not support multi-GPU training"
        dev = torch.device('cuda:0')
        gpus = [torch.device('cuda:0')] if backend.comm.size() == 1 else [torch.device('cuda:'+str(i)) for i in range(1, backend.comm.size())]
        optimizer = MFAC(model.parameters(), 
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay, 
                ngrads=32, 
                moddev=dev,
                optdev=dev,
                gpus=gpus)
    elif args.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    if args.use_kfac:
        KFAC = kfac.get_kfac_module(args.kfac_name)
        preconditioner = KFAC(model, 
                lr=args.base_lr, 
                factor_decay=args.stat_decay, 
                damping=args.damping, 
                kl_clip=args.kl_clip, 
                fac_update_freq=args.kfac_cov_update_freq, 
                kfac_update_freq=args.kfac_update_freq, 
                exclude_parts=args.exclude_parts)
        #kfac_param_scheduler = kfac.KFACParamScheduler(
        #        preconditioner,
        #        damping_alpha=1,
        #        damping_schedule=None,
        #        update_freq_alpha=1,
        #        update_freq_schedule=None)
    else:
        preconditioner = None

    # Distributed Optimizer
    if args.horovod:
        import horovod.torch as hvd
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer, 
                                            named_parameters=model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Average,
                                            backward_passes_per_step=1)
        if hvd.size() > 1:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Learning Rate Schedule
    if args.lr_schedule == 'cosine':
        lrs = create_cosine_lr_schedule(args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch)
    elif args.lr_schedule == 'polynomial':
        lrs = create_polynomial_lr_schedule(args.base_lr, args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch, lr_end=0.0, power=2.0)
    elif args.lr_schedule == 'step':
        lrs = create_multi_step_lr_schedule(backend.comm.size(), args.warmup_epochs, args.lr_decay, args.lr_decay_alpha)
    
    lr_scheduler = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_scheduler.append(LambdaLR(preconditioner, lrs)) # lr schedule for preconditioner as well
        #lr_scheduler.append(kfac_param_scheduler)

    return model, optimizer, preconditioner, lr_scheduler, criterion

def train(epoch, model, optimizer, preconditioner, lr_scheduler, criterion, train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)

    train_loss = Metric('train_loss')
    display = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, data) # MSE
        
        with torch.no_grad():
            train_loss.update(loss)

        loss.backward()

        if args.horovod:
            optimizer.synchronize()

        if args.use_kfac:
            preconditioner.step(epoch=epoch)
    
        if args.horovod:
            with optimizer.skip_synchronize():
                optimizer.step()
        else:
            optimizer.step()
        avg_time += (time.time()-stime)
            
        if (batch_idx + 1) % display == 0:
            if args.verbose:
                logger.info("[%d][%d] train loss: %.4f" % (epoch, batch_idx, train_loss.avg.item()))

        if not args.lr_schedule == 'step':
            for scheduler in lr_scheduler:
                scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch train loss: %.4f" % (epoch, train_loss.avg.item()))
        if wandb:
            wandb.log({"train loss": train_loss.avg.item()})

    if args.lr_schedule == 'step':
        for scheduler in lr_scheduler:
            scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch learning rate: %f" % (epoch, optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, _, test_loader = get_dataset(args)
    args.num_steps_per_epoch = len(train_loader)
    model, optimizer, preconditioner, lr_scheduler, criterion = get_model(args)

    start = time.time()

    for epoch in range(args.epochs):
        stime = time.time()
        train(epoch, model, optimizer, preconditioner, lr_scheduler, criterion, train_sampler, train_loader, args)
        if args.verbose:
            logger.info("[%d] epoch train time: %.3f"%(epoch, time.time() - stime))

    if args.verbose:
        logger.info("Total Training Time: %s", str(datetime.timedelta(seconds=time.time() - start)))

