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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import torch.multiprocessing as mp

import cifar_resnet as resnet
from cifar_wide_resnet import Wide_ResNet
from cifar_pyramidnet import ShakePyramidNet
from cifar_vgg import VGG
from vit import VisionTransformer, ViT_CONFIGS  
from efficientnet import EfficientNet
from utils import *

import kfac
import kfac.backend as backend #don't use a `from` import
from shampoo.shampoo import Shampoo
from mfac.optim import MFAC

import horovod.torch as hvd
import torch.distributed as dist

import wandb
wandb = False
SPEED = False

def initialize():
    # Training Parameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--model', type=str, default='resnet32',
                        help='ResNet model to use [20, 32, 56]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 or cifar100')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
                        choices=['step', 'linear', 'polynomial', 'cosine'], help='learning rate schedules')
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
    parser.add_argument('--mixup', type=int, default=0, metavar='WE',
                        help='use mixup (default: 0)')
    parser.add_argument('--cutmix', type=int, default=0, metavar='WE',
                        help='use cutmix (default: 0)')
    parser.add_argument('--autoaugment', type=int, default=0, metavar='WE',
                        help='use autoaugment (default: 0)')
    parser.add_argument('--cutout', type=int, default=0, metavar='WE',
                        help='use cutout augment (default: 0)')
    parser.add_argument('--opt-name', type=str, default='sgd', metavar='WE',
                        help='choose base optimizer [sgd, adam, shampoo, mfac]')
    parser.add_argument('--use-pretrained-model', type=int, default=0, metavar='WE',
                        help='use pretrained model e.g. ViT-B_16 (default: 0)')
    parser.add_argument('--pretrained-dir', type=str, default='/datasets/pretrained_models/',
                        help='pretrained model dir')

    # KFAC Parameters
    parser.add_argument('--kfac-type', type=str, default='Femp', 
                        help='choices: F1mc or Femp') 
    parser.add_argument('--kfac-name', type=str, default='inverse',
                        help='choices: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
                        help='choices: ComputeFactor,CommunicateFactor,ComputeInverse,CommunicateInverse')
    parser.add_argument('--kfac-update-freq', type=int, default=1,
                        help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.03,
                        help='KFAC damping factor (defaultL 0.03)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')

    # Other Parameters
    parser.add_argument('--log-dir', default='./logs',
                        help='log directory')
    parser.add_argument('--dir', type=str, default='/datasets/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
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
        '{}_{}_ep{}_bs{}_lr{}_gpu{}_kfac{}_{}_{}_momentum{}_damping{}_stat{}_clip{}.log'.format(args.dataset, args.model, args.epochs, args.batch_size, args.base_lr, backend.comm.size(), args.kfac_update_freq, algo, args.lr_schedule, args.momentum, args.damping, args.stat_decay, args.kl_clip))
        #'{}_{}_ep{}_bs{}_gpu{}_kfac{}_{}_{}_lr{}_seed{}.log'.format(args.dataset, args.model, args.epochs, args.batch_size, backend.comm.size(), args.kfac_update_freq, algo, args.lr_schedule, args.base_lr, args.seed))

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

    pretrained_model = args.model.lower() if args.use_pretrained_model else None
    transform_train, transform_test = get_transform(args.dataset, 
            args.autoaugment, args.cutout, pretrained_model)
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.dir, train=True, 
                                     download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=args.dir, train=False,
                                    download=False, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=args.dir, train=True, 
                                     download=False, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=args.dir, train=False,
                                    download=False, transform=transform_test)


    # Use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    #train_loader = torch.utils.data.DataLoader(train_dataset,
    train_loader = MultiEpochsDataLoader(train_dataset,
            batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # Use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    #test_loader = torch.utils.data.DataLoader(test_dataset, 
    test_loader = MultiEpochsDataLoader(test_dataset, 
            batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)
    
    return train_sampler, train_loader, test_sampler, test_loader


def get_model(args):
    num_classes = 10 if args.dataset == 'cifar10' else 100
    # ResNet
    if args.model.lower() == "resnet20":
        model = resnet.resnet20(num_classes=num_classes)
    elif args.model.lower() == "resnet32":
        model = resnet.resnet32(num_classes=num_classes)
    elif args.model.lower() == "resnet44":
        model = resnet.resnet44(num_classes=num_classes)
    elif args.model.lower() == "resnet56":
        model = resnet.resnet56(num_classes=num_classes)
    elif args.model.lower() == "resnet110":
        model = resnet.resnet110(num_classes=num_classes)
    elif args.model.lower() == "wrn28-10":
        model = Wide_ResNet(28, 10, 0.0, num_classes=num_classes)
    elif args.model.lower() == "wrn28-20":
        model = Wide_ResNet(28, 20, 0.0, num_classes=num_classes)
    elif args.model.lower() == "pyramidnet":
        model = ShakePyramidNet(depth=110, alpha=270, num_classes=num_classes)
    elif args.model.lower() == "vgg16":
        model = VGG("VGG16", num_classes=num_classes)
    elif args.model.lower() == "vgg19":
        model = VGG("VGG19", num_classes=num_classes)
    elif args.model.lower() == "vit-b16" and args.use_pretrained_model:
        vit_config = ViT_CONFIGS[ "vit-b16"]
        model = VisionTransformer(vit_config, img_size=224, zero_head=True, num_classes=num_classes)
        model.load_from(np.load(args.pretrained_dir + "ViT-B_16.npz"))
    elif args.model.lower() == "vit-b16" and not args.use_pretrained_model:
        vit_config = ViT_CONFIGS[ "vit-b16"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif args.model.lower() == "vit-s8":
        vit_config = ViT_CONFIGS[ "vit-s8"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif args.model.lower() == "vit-t8":
        vit_config = ViT_CONFIGS[ "vit-t8"]
        model = VisionTransformer(vit_config, img_size=32, num_classes=num_classes)
    elif "efficientnet" in args.model.lower() and args.use_pretrained_model:
        model = EfficientNet.from_pretrained(args.model.lower(), 
                weights_path=args.pretrained_dir + args.model.lower() + ".pth",
                num_classes=num_classes)


    if args.cuda:
        model.cuda()

    # Optimizer
    if args.label_smoothing:
        criterion = LabelSmoothLoss(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

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
    elif args.lr_schedule == 'linear':
        lrs = create_polynomial_lr_schedule(args.base_lr, args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch, lr_end=0.0, power=1.0)
    elif args.lr_schedule == 'polynomial':
        lrs = create_polynomial_lr_schedule(args.base_lr, args.warmup_epochs * args.num_steps_per_epoch, args.epochs * args.num_steps_per_epoch, lr_end=0.0, power=2.0)
    elif args.lr_schedule == 'step':
        #lrs = create_multi_step_lr_schedule(backend.comm.size(), args.warmup_epochs, args.lr_decay, args.lr_decay_alpha)
        lrs = create_multi_step_lr_schedule(max(4, backend.comm.size()), args.warmup_epochs, args.lr_decay, args.lr_decay_alpha)
    
    lr_scheduler = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_scheduler.append(LambdaLR(preconditioner, lrs)) # lr schedule for preconditioner as well
        #lr_scheduler.append(kfac_param_scheduler)

    return model, optimizer, preconditioner, lr_scheduler, criterion

def train(epoch, model, optimizer, preconditioner, lr_scheduler, criterion, train_sampler, train_loader, args):
    model.train()
    train_sampler.set_epoch(epoch)
    if args.cutmix:
        cutmix = CutMix(size=32, beta=1.0)
    elif args.mixup:
        mixup = MixUp(alpha=1.0)
    
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    avg_time = 0.0
    display = 10
    iotimes=[];fwbwtimes=[];kfactimes=[];commtimes=[];uptimes=[]
    ittimes=[]

    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        if args.cutmix:
            data, target, rand_target, lambda_ = cutmix((data, target))
        elif args.mixup:
            if np.random.rand() <= 0.8:
                data, target, rand_target, lambda_ = mixup((data, target))
            else:
                rand_target, lambda_ = torch.zeros_like(target), 1.0

        optimizer.zero_grad()
        iotime = time.time()
        iotimes.append(iotime-stime)

        output = model(data)
        if type(output) == tuple:
            output = output[0]

        if args.cutmix or args.mixup:
            loss = criterion(output, target) * lambda_ + criterion(output, rand_target) * (1.0 - lambda_)
        else:
            loss = criterion(output, target)
        
        with torch.no_grad():
            train_loss.update(loss)
            train_accuracy.update(accuracy(output, target))

        loss.backward()

        fwbwtime = time.time()
        fwbwtimes.append(fwbwtime-iotime)

        if args.horovod:
            optimizer.synchronize()
        commtime = time.time()
        commtimes.append(commtime-fwbwtime)
        
        if args.use_kfac:
            preconditioner.step(epoch=epoch)
            
        kfactime = time.time()
        kfactimes.append(kfactime-commtime)
    
        if args.horovod:
            with optimizer.skip_synchronize():
                optimizer.step()
        else:
            optimizer.step()

        updatetime=time.time()
        uptimes.append(updatetime-kfactime)
        avg_time += (time.time()-stime)
            
        if (batch_idx + 1) % display == 0:
            if args.verbose:
                logger.info("[%d][%d] train loss: %.4f, acc: %.3f" % (epoch, batch_idx, train_loss.avg.item(), 100*train_accuracy.avg.item()))

            if args.verbose and SPEED:
                logger.info("[%d][%d] time: %.3f, speed: %.3f images/s" % (epoch, batch_idx, avg_time/display, args.batch_size/(avg_time/display)))
                logger.info('Profiling: IO: %.3f, FW+BW: %.3f, COMM: %.3f, KFAC: %.3f, UPDAT: %.3f', np.mean(iotimes), np.mean(fwbwtimes), np.mean(commtimes), np.mean(kfactimes), np.mean(uptimes))
            iotimes=[];fwbwtimes=[];kfactimes=[];commtimes=[];uptimes=[]
            ittimes.append(avg_time/display)
            avg_time = 0.0

        if batch_idx >= (display * 6) and SPEED:
            if args.verbose:
                logger.info("Iteration time: mean %.3f, std: %.3f" % (np.mean(ittimes[1:]),np.std(ittimes[1:])))
            break

        if not args.lr_schedule == 'step':
            for scheduler in lr_scheduler:
                scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))
        if wandb:
            wandb.log({"train loss": train_loss.avg.item(), "train acc": train_accuracy.avg.item()})

    if args.lr_schedule == 'step':
        for scheduler in lr_scheduler:
            scheduler.step()

    if args.verbose:
        logger.info("[%d] epoch learning rate: %f" % (epoch, optimizer.param_groups[0]['lr']))


def test(epoch, model, criterion, test_loader, args):
    model.eval()
    test_loss = Metric('val_loss')
    test_accuracy = Metric('val_accuracy')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if type(output) == tuple:
                output = output[0]
            test_loss.update(criterion(output, target))
            test_accuracy.update(accuracy(output, target))
            
    if args.verbose:
        logger.info("[%d] evaluation loss: %.4f, acc: %.3f" % (epoch, test_loss.avg.item(), 100*test_accuracy.avg.item()))
        if wandb:
            wandb.log({"eval loss": test_loss.avg.item(), "eval acc": test_accuracy.avg.item()})


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.set_start_method('forkserver')

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
        if not SPEED:
            test(epoch, model, criterion, test_loader, args)

    if args.verbose:
        logger.info("Total Training Time: %s", str(datetime.timedelta(seconds=time.time() - start)))

