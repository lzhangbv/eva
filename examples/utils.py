import torch
import torch.nn.functional as F
# import horovod.torch as hvd
import kfac.backend as backend

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
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

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

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
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
