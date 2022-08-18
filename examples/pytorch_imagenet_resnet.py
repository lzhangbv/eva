from __future__ import print_function

import time
from datetime import datetime, timedelta
import argparse
import os
import math
import sys
import warnings
import numpy as np
from distutils.version import LooseVersion
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
strhdlr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

SPEED = True

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision import datasets, transforms
import horovod.torch as hvd
import torch.distributed as dist
from tqdm import tqdm
from distutils.version import LooseVersion
import imagenet_resnet as models
import imagenet_inceptionv4 as inceptionv4
from vit import VisionTransformer, ViT_CONFIGS  
from efficientnet import EfficientNet
from utils import *

import kfac
import kfac.backend as backend #don't use a `from` import
from shampoo.shampoo import Shampoo
from mfac.optim import MFAC

#os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '10' 

STEP_FIRST = LooseVersion(torch.__version__) < LooseVersion('1.1.0')

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def initialize():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--horovod', type=int, default=1, metavar='N',
                        help='whether use horovod as communication backend (default: 1)')
    parser.add_argument('--train-dir', default='/tmp/imagenet/ILSVRC2012_img_train/',
                        help='path to training data')
    parser.add_argument('--val-dir', default='/tmp/imagenet/ILSVRC2012_img_val/',
                        help='path to validation data')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard/checkpoint log directory')
    parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--model', default='resnet50',
                        help='Model (resnet35, resnet50, resnet101, resnet152, resnext50, resnext101)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[30, 60, 80],
                        help='epoch intervals to decay lr')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.00005,
                        help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing (default 0.1)')
    parser.add_argument('--opt-name', type=str, default='sgd', metavar='WE',
                        help='choose base optimizer [sgd, adam, shampoo, mfac]')
    parser.add_argument('--ngrads', type=int, default=32, metavar='WE',
                        help='number of gradients used to estimate FIM in M-FAC')

    # KFAC Parameters
    parser.add_argument('--kfac-name', type=str, default='inverse',
            help='choises: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
            help='choises: CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor')
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 = no kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq schedule (default None)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.03,
                        help='KFAC damping factor (default 0.03)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=[40, 80],
                        help='KFAC damping decay schedule (default [40, 80])')
    parser.add_argument('--kl-clip', type=float, default=0.01,
                        help='KL clip (default: 0.01)')
    parser.add_argument('--diag-blocks', type=int, default=1,
                        help='Number of blocks to approx layer factor with (default: 1)')
    parser.add_argument('--diag-warmup', type=int, default=0,
                        help='Epoch to start diag block approximation at (default: 0)')
    parser.add_argument('--distribute-layer-factors', action='store_true', default=None,
                        help='Compute A and G for a single layer on different workers. '
                              'None to determine automatically based on worker and '
                              'layer count.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--single-threaded', action='store_true', default=False,
                        help='disables multi-threaded dataloading')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_kfac = True if (args.opt_name == 'sgd' and args.kfac_update_freq > 0) else False

    # Comm backend init
    if args.horovod:
        hvd.init()
        backend.init("Horovod")
    else:
        dist.init_process_group(backend='nccl', init_method='env://')
        backend.init("Torch")

    args.local_rank = backend.comm.local_rank()

    torch.manual_seed(args.seed)
    args.verbose = 1 if backend.comm.rank() == 0 else 0
    #if args.verbose:
    #    logger.info(args)

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    args.log_dir = os.path.join(args.log_dir, 
                                "imagenet_resnet50_kfac{}_gpu_{}_{}".format(
                                args.kfac_update_freq, backend.comm.size(),
                                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    args.checkpoint_format=os.path.join(args.log_dir, args.checkpoint_format)
    #os.makedirs(args.log_dir, exist_ok=True)

    # If set > 0, will resume training from a given checkpoint.
    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break

    # Horovod: write TensorBoard logs on first worker.
    try:
        if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter
        #args.log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
        args.log_writer = None
    except ImportError:
        args.log_writer = None
    
    algo = args.kfac_name if args.use_kfac else args.opt_name
    logfile = './logs/imagenet_{}_bs{}_gpu{}_kfac{}_{}.log'.format(args.model, args.batch_size, backend.comm.size(), args.kfac_update_freq, algo)
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    if args.verbose:
        logger.info(args)

    return args

def get_datasets(args):
    # Horovod: limit # of CPU threads to be used per worker.
    if args.single_threaded:
        torch.set_num_threads(4)
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    else:
        torch.set_num_threads(8)
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
    val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size * args.batches_per_allreduce,
            sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=backend.comm.size(), rank=backend.comm.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size,
            sampler=val_sampler, **kwargs)

    return train_sampler, train_loader, val_sampler, val_loader

def get_model(args):
    if args.model.lower() == 'resnet18':
        model = models.resnet18()
    elif args.model.lower() == 'resnet34':
        model = models.resnet34()
    elif args.model.lower() == 'resnet50':
        model = models.resnet50()
    elif args.model.lower() == 'resnet101':
        model = models.resnet101()
    elif args.model.lower() == 'resnet152':
        model = models.resnet152()
    elif args.model.lower() == 'resnext50':
        model = models.resnext50_32x4d()
    elif args.model.lower() == 'resnext101':
        model = models.resnext101_32x8d()
    elif args.model.lower() == 'densenet121':
        model = torchvision.models.densenet121(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'densenet201':
        model = torchvision.models.densenet201(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'vgg11':
        model = torchvision.models.vgg11(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'vgg16':
        model = torchvision.models.vgg16(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'vgg19':
        model = torchvision.models.vgg19(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'inceptionv3':
        model = torchvision.models.inception_v3(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'inceptionv4':
        model = inceptionv4.inceptionv4(num_classes=1000,pretrained=False)
    elif args.model.lower() == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2()
    elif 'efficientnet' in args.model.lower():
        model = EfficientNet.from_name(args.model.lower())
    elif args.model.lower() == "vit-b16":
        vit_config = ViT_CONFIGS[ "vit-b16"]
        model = VisionTransformer(vit_config, img_size=224, zero_head=True, num_classes=1000)
    elif args.model.lower() == "vit-l16":
        vit_config = ViT_CONFIGS[ "vit-l16"]
        model = VisionTransformer(vit_config, img_size=224, zero_head=True, num_classes=1000)
    else:
        raise ValueError('Unknown model \'{}\''.format(args.model))

    if args.cuda:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    args.base_lr = args.base_lr * backend.comm.size() * args.batches_per_allreduce
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
        dev = torch.device('cuda')
        gpus = [dev]
        optimizer = MFAC(model.parameters(), 
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay, 
                ngrads=args.ngrads, 
                moddev=dev,
                optdev=dev,
                gpus=gpus)
    elif args.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    if args.use_kfac > 0:
        KFAC = kfac.get_kfac_module(args.kfac_name)
        preconditioner = KFAC(
                model, lr=args.base_lr, factor_decay=args.stat_decay,
                damping=args.damping, kl_clip=args.kl_clip,
                fac_update_freq=args.kfac_cov_update_freq,
                kfac_update_freq=args.kfac_update_freq,
                #diag_blocks=args.diag_blocks,
                #diag_warmup=args.diag_warmup,
                #distribute_layer_factors=args.distribute_layer_factors, 
                exclude_parts=args.exclude_parts)
        kfac_param_scheduler = kfac.KFACParamScheduler(
                preconditioner,
                damping_alpha=args.damping_alpha,
                damping_schedule=args.damping_decay,
                update_freq_alpha=args.kfac_update_freq_alpha,
                update_freq_schedule=args.kfac_update_freq_decay,
                start_epoch=args.resume_from_epoch)
    else:
        preconditioner = None

    if args.horovod:
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters(),
                compression=compression, op=hvd.Average,
                backward_passes_per_step=args.batches_per_allreduce)
    
        # Horovod: broadcast parameters & optimizer state.
        if hvd.size() > 1:
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank]) 

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights 
    # to other workers.
    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    lrs = create_multi_step_lr_schedule(backend.comm.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [LambdaLR(optimizer, lrs)]
    if preconditioner is not None:
        lr_scheduler.append(LambdaLR(preconditioner, lrs))
        lr_scheduler.append(kfac_param_scheduler)

    loss_func = LabelSmoothLoss(args.label_smoothing)

    return model, optimizer, preconditioner, lr_scheduler, lrs, loss_func

def train(epoch, model, optimizer, preconditioner, lr_schedules, lrs,
          loss_func, train_sampler, train_loader, args):

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    avg_time = 0.0
    display = 50

    if STEP_FIRST:
        for scheduler in lr_schedules:
            scheduler.step()

    #with tqdm(total=len(train_loader), 
    #          desc='Epoch {:3d}/{:3d}'.format(epoch + 1, args.epochs),
    #          disable=not args.verbose) as t:
    profiling=True
    iotimes = [];fwbwtimes=[];kfactimes=[];commtimes=[];uptimes=[]
    ittimes = []
    if True:
        for batch_idx, (data, target) in enumerate(train_loader):
            stime = time.time()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            iotime = time.time()
            iotimes.append(iotime-stime)

            optimizer.zero_grad()
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                if type(output) == tuple:
                    output = output[0]
                
                loss = loss_func(output, target_batch)

                with torch.no_grad():
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(output, target_batch))

                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()        
            fwbwtime = time.time()
            fwbwtimes.append(fwbwtime-iotime)

            if args.horovod:
                optimizer.synchronize()
            commtime = time.time()
            commtimes.append(commtime-fwbwtime)
            if preconditioner is not None:
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

            if batch_idx > 0 and batch_idx % display == 0:
                if args.verbose and SPEED:
                    logger.info("[%d][%d] loss: %.4f, acc: %.2f, time: %.3f, speed: %.3f images/s" % (epoch, batch_idx, train_loss.avg.item(), 100*train_accuracy.avg.item(), avg_time/display, args.batch_size/(avg_time/display)))
                    logger.info('Profiling: IO: %.3f, FW+BW: %.3f, COMM: %.3f, KFAC: %.3f, STEP: %.3f', np.mean(iotimes), np.mean(fwbwtimes), np.mean(commtimes), np.mean(kfactimes), np.mean(uptimes))
                    iotimes = [];fwbwtimes=[];kfactimes=[];commtimes=[]
                ittimes.append(avg_time/display)
                avg_time = 0.0
            if batch_idx > 6 * display and SPEED:
                if args.verbose:
                    logger.info("Iteration time: mean %.3f, std: %.3f" % (np.mean(ittimes[1:]),np.std(ittimes[1:])))
                    #logger.info("Max memory allocated %.1f MB" % (torch.cuda.max_memory_allocated()/1024/1024))
                break
        if args.verbose:
            logger.info("[%d] epoch train loss: %.4f, acc: %.3f" % (epoch, train_loss.avg.item(), 100*train_accuracy.avg.item()))

    if not STEP_FIRST:
        for scheduler in lr_schedules:
            scheduler.step()

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        args.log_writer.add_scalar('train/lr', args.base_lr * lrs(epoch), epoch)

def validate(epoch, model, loss_func, val_loader, args):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    #with tqdm(total=len(val_loader),
    #          bar_format='{l_bar}{bar}|{postfix}',
    #          desc='             '.format(epoch + 1, args.epochs),
    #          disable=not args.verbose) as t:
    if True:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))
            if args.verbose:
                logger.info("[%d][0] evaluation loss: %.4f, acc: %.3f" % (epoch, val_loss.avg.item(), 100*val_accuracy.avg.item()))

                #t.update(1)
                #if i + 1 == len(val_loader):
                #    t.set_postfix_str("\b\b val_loss: {:.4f}, val_acc: {:.2f}%".format(
                #            val_loss.avg.item(), 100*val_accuracy.avg.item()),
                #            refresh=False)

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()

    train_sampler, train_loader, _, val_loader = get_datasets(args)
    model, opt, preconditioner, lr_schedules, lrs, loss_func = get_model(args)

    if args.verbose:
        logger.info("MODEL: %s", args.model)

    start = time.time()

    for epoch in range(args.resume_from_epoch, args.epochs):
        train(epoch, model, opt, preconditioner, lr_schedules, lrs,
             loss_func, train_sampler, train_loader, args)
        if not SPEED:
            validate(epoch, model, loss_func, val_loader, args)
            #save_checkpoint(model, opt, args.checkpoint_format, epoch)

    if args.verbose:
        logger.info("\nTraining time: %s", str(timedelta(seconds=time.time() - start)))
