import torch
import torch.nn.functional as F
import numpy as np
import math
# import horovod.torch as hvd
import kfac.backend as backend
import torchvision.transforms as transforms
from autoaugment import CIFAR10Policy, SVHNPolicy

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()

def save_checkpoint(model, optimizer, checkpoint_format, epoch):
    if backend.comm.rank() == 0:
        filepath = checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

class LabelSmoothLoss(torch.nn.Module):
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# Code: https://github.com/facebookresearch/mixup-cifar10
class MixUp(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        x, y = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CutMix(object):
    def __init__(self, size, beta):
        self.size = size
        self.beta = beta

    def __call__(self, batch):
        img, label = batch
        rand_img, rand_label = self._shuffle_minibatch(batch)
        lambda_ = np.random.beta(self.beta,self.beta)
        r_x = np.random.uniform(0, self.size)
        r_y = np.random.uniform(0, self.size)
        r_w = self.size * np.sqrt(1-lambda_)
        r_h = self.size * np.sqrt(1-lambda_)
        x1 = int(np.clip(r_x - r_w // 2, a_min=0, a_max=self.size))
        x2 = int(np.clip(r_x + r_w // 2, a_min=0, a_max=self.size))
        y1 = int(np.clip(r_y - r_h // 2, a_min=0, a_max=self.size))
        y2 = int(np.clip(r_y + r_h // 2, a_min=0, a_max=self.size))
        img[:, :, x1:x2, y1:y2] = rand_img[:, :, x1:x2, y1:y2]
        
        lambda_ = 1 - (x2-x1)*(y2-y1)/(self.size*self.size)
        return img, label, rand_label, lambda_
    
    def _shuffle_minibatch(self, batch):
        img, label = batch
        rand_img, rand_label = img.clone(), label.clone()
        rand_idx = torch.randperm(img.size(0))
        rand_img, rand_label = rand_img[rand_idx], rand_label[rand_idx]
        return rand_img, rand_label

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def get_transform(dataset, autoaugment=0, cutout=0, pretrain_model=None):
    if dataset == 'cifar10':
        num_classes = 10
        size = 32
        padding = 4
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif dataset == 'cifar100':
        num_classes = 100
        size = 32
        padding = 4
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif dataset == "svhn":
        num_classes = 10
        size = 32
        padding = 4
        mean, std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
    
    if pretrain_model is not None:
        if 'vit' in pretrain_model:
            train_transform = [transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0))]
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        elif 'efficientnet' in pretrain_model:
            train_transform = [transforms.RandomResizedCrop(224)]
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        train_transform = [transforms.RandomCrop(size=size, padding=padding)]
    if dataset != "svhn":
        train_transform += [transforms.RandomHorizontalFlip()]
    if autoaugment:
        if dataset == "cifar10" or "cifar100":
            train_transform.append(CIFAR10Policy())
        elif dataset == 'svhn':
            train_transform.append(SVHNPolicy())
    train_transform += [transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)]
    if cutout:
        if dataset == "cifar10":
            train_transform.append(Cutout(n_holes=1, length=16))
        elif dataset == "cifar100":
            train_transform.append(Cutout(n_holes=1, length=8))
        elif dataset == "svhn":
            train_transform.append(Cutout(n_holes=1, length=20))    

    train_transform = transforms.Compose(train_transform)

    if pretrain_model is not None:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)])
    return train_transform, test_transform

def metric_average(val_tensor):
    backend.comm.allreduce(val_tensor)
    return val_tensor.item()

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val, n=1):
        backend.comm.allreduce(val, name=self.name)
        self.sum += float(val)
        self.n += n

    @property
    def avg(self):
        return self.sum / self.n

def create_multi_step_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule

def create_polynomial_lr_schedule(lr_init, num_warmup_steps, num_training_steps, lr_end=0.0, power=1.0):
    def lr_schedule(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init
    return lr_schedule

def create_cosine_lr_schedule(num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_schedule(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return lr_schedule

# F1mc: sample pseudo labels from model distributions
def generate_pseudo_labels(outputs):
    """
    Args:
    outputs: model outputs before softmax layer for multi-classification
    """
    dist = F.softmax(outputs, dim=1)
    pseudo_labels = torch.multinomial(dist, num_samples=1).view(-1)
    return pseudo_labels


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
                                                    
    def __len__(self):
        return len(self.batch_sampler.sampler)
                                                                
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
                                                                                                
                                                                                                
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
                                                                                                                        
    def __init__(self, sampler):
        self.sampler = sampler
                                                                                                                                        
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
