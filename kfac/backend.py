import torch
import horovod.torch as hvd
import torch.distributed as dist
import os
import enum


"""
Collective Communication Backend

Usage:
    import kfac.backend as backend
    
    hvd.init() or dist.init()
    backend.init()
    backend.comm.APIs()
"""


# global comm object
comm = None

# communicate operations
class Ops(enum.Enum):
    Average = "average"
    Sum = "sum"

# init backend
def init(backend):
    global comm
    if comm is None:
        comm = _get_comm_backend(backend)

def _get_comm_backend(backend):
        if backend == "Horovod":
            try:
                hvd.size()
                return _HorovodBackend()
            except:
                return RuntimeError('Horovod much be init before create HorovodBackend.')
        elif backend == "Torch":
            try:
                dist.get_world_size()
                return _TorchBackend()
            except:
                return RuntimeError('Torch.distributed much be init before create TorchBackend.')
        else:
            return RuntimeError('The backend is not implemented. Now only Horovod and Torch are supported.')


class _HorovodBackend:
    """
    Collective communication backend based on Horovod
    """
    def __init__(self):
        self.Average = Ops.Average
        self.Sum = Ops.Sum

    def size(self):
        return hvd.size()

    def local_rank(self):
        return hvd.local_rank()

    def rank(self):
        return hvd.rank()
    
    def new_group(self, ranks): # support process_sets after v0.23.0
        return hvd.add_process_set(ranks)

    def _get_op(self, op):
        if op == Ops.Average:
            return hvd.Average
        elif op == Ops.Sum:
            return hvd.Sum
        else:
            raise ValueError('Unknown communication operation {}'.format(op))
    
    def allreduce(self, tensor, name=None, op=Ops.Average):
        self.allreduce_(tensor, name, op)

    def allreduce_(self, tensor, name=None, op=Ops.Average):
        op = self._get_op(op)
        hvd.allreduce_(tensor, name=name, op=op) # in-place synchronous all-reduce

    def allreduce_async_(self, tensor, name=None, op=Ops.Average):
        op = self._get_op(op)
        return hvd.allreduce_async_(tensor, name=name, op=op) # in-place asynchronous all-reduce

    def broadcast(self, tensor, src, group=None, name=None):
        self.broadcast_(tensor, src, group, name)
    
    def broadcast_(self, tensor, src, group=None, name=None):
        if group is None:
            hvd.broadcast_(tensor, root_rank=src, name=name) # in-place synchronous broadcast
        else:
            hvd.broadcast_(tensor, root_rank=src, process_set=group, name=name)
    
    def broadcast_async_(self, tensor, src, group=None, name=None): # in-place asynchronous broadcast
        if group is None:
            return hvd.broadcast_async_(tensor, root_rank=src, name=name)
        else:
            return hvd.broadcast_async_(tensor, root_rank=src, process_set=group, name=name)

    def synchronize(self, handle):
        return hvd.synchronize(handle)



class _TorchBackend:
    """
    Collective communication backend based on Pytorch DDP
    """
    def __init__(self):
        self.Average = Ops.Average
        self.Sum = Ops.Sum

    def size(self):
        return dist.get_world_size()

    def local_rank(self):
        try:
            return int(os.environ['LOCAL_RANK'])
        except:
            raise RuntimeError('LOCAL_RANK must be set in the environment when using torch.distributed')

    def rank(self):
        return dist.get_rank()

    def new_group(self, ranks):
        return dist.new_group(ranks)
        
    def allreduce(self, tensor, name=None, op=Ops.Average):
        self.allreduce_(tensor, name, op)

    def allreduce_(self, tensor, name=None, op=Ops.Average):
        dist.all_reduce(tensor, async_op=False)
        if op == Ops.Average:
            tensor.div_(self.size())

    def allreduce_async_(self, tensor, name=None, op=Ops.Average):
        handle = dist.all_reduce(tensor, async_op=True)
        if op == Ops.Sum:
            return handle
        else:
            return (handle, tensor) # wait to be averaged

    def broadcast(self, tensor, src, group=None, name=None):
        self.broadcast_(tensor, src, group, name)
    
    def broadcast_(self, tensor, src, group=None, name=None):
        dist.broadcast(tensor, src=src, group=group, async_op=False)
    
    def broadcast_async_(self, tensor, src, group=None, name=None):
        return dist.broadcast(tensor, src=src, group=group, async_op=True)

    def synchronize(self, handle):
        if isinstance(handle, tuple):
            h, tensor = handle
            h.wait()
            tensor.div_(self.size())
        else:
            handle.wait()
        return hvd.synchronize(handle)


