import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from kfac.utils import get_factor_A, get_factor_G, mat_inv
import logging
logger = logging.getLogger()


class KFAC(optim.Optimizer):
    """Distributed K-FAC that communicates KFs (no MP). 
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling
      factor_decay (float): running average coefficient for KVs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 damping=0.03,
                 fac_update_freq=1,  
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 diag_blocks=4,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.fac_update_freq = fac_update_freq      # freq for computing KFs
        self.kfac_update_freq = kfac_update_freq    # freq for communicating and inverting KFs
        self.kfac_batch_size = kfac_batch_size      # subsampled batch size for computing KFs
        self.diag_blocks = diag_blocks              # inverting KFs blocks with matrix split
        self.kl_clip = kl_clip if kl_clip > 0 else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_A, self.m_G = {}, {}
        self.m_inv_A, self.m_inv_G = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None
        self.steps = 0

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving KFs (A)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_factor_A(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_A:
                    self.m_A[module] = new
                else:
                    #self.m_A[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_A[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
            if backend.comm.size() > 1 and self.steps % self.kfac_update_freq == 0:
                self.handles.append(backend.comm.allreduce_async_(self.m_A[module], op=backend.comm.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving KFs (G)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_factor_G(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_G:
                    self.m_G[module] = new
                else:
                    #self.m_G[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_G[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
            if backend.comm.size() > 1 and self.steps % self.kfac_update_freq == 0:
                self.handles.append(backend.comm.allreduce_async_(self.m_G[module], op=backend.comm.Average))

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                #module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        if backend.comm.rank() == 0:
            logger.info("#register modules: %s", len(self.modules))

    ### Invert KFs
    def _compute_inverse(self):
        for module in self.modules:
            A = self.m_A[module]
            G = self.m_G[module]

            # initialize the memory
            if module not in self.m_inv_A:
                self.m_inv_A[module] = A.new_zeros(A.shape)
            if module not in self.m_inv_G:
                self.m_inv_G[module] = G.new_zeros(G.shape)

            # scaling the damping value
            # pi = 1
            pi = torch.sqrt((A.trace()/A.shape[0])/(G.trace()/G.shape[0]))
            
            # invert with diag blocks
            self._invert_diag_blocks(A, self.m_inv_A[module], damping=(self.damping ** 0.5) * pi)
            self._invert_diag_blocks(G, self.m_inv_G[module], damping=(self.damping ** 0.5) / pi)
            

    def _invert_diag_blocks(self, KF, inv_KF, damping):
        """invert diag block approximated matrix"""
        Ntotal = KF.shape[0]
        Nsections = min(self.diag_blocks, Ntotal)
        div_points = self._get_div_points(Ntotal, Nsections)
        for i in range(Nsections):
            st = div_points[i]
            end = div_points[i + 1]
            block = KF[st:end, st:end]
            block.add_(torch.diag(block.new(block.shape[0]).fill_(damping)))
            inverse = mat_inv(block)
            inv_KF.data[st:end, st:end].copy_(inverse)

    def _get_div_points(self, Ntotal, Nsections):
        """compute div_points to split Ntotal elements into Nsection blocks almost equally"""
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section+1] + (Nsections-extras) * [Neach_section])
        return np.cumsum(section_sizes)


	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via K-FAC"""
        vg_sum = 0
        for module in self.modules:
            # compute preconditioned grads
            grad = self._get_grad(module)
            v = self.m_inv_G[module] @ grad @ self.m_inv_A[module]

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # kl clip: grad and precon grad
                if self.kl_clip is not None:
                    vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                module.weight.grad.data.copy_(weight)
            del v

        # kl clip
        if self.kl_clip is not None:
            nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad    


    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        if self.steps % self.kfac_update_freq == 0:
            if backend.comm.size() > 1:
                for handle in self.handles:
                    backend.comm.synchronize(handle)
                self.handles = []
            self._compute_inverse()
        
        self._precondition_grads()

        self.steps += 1

class KFACParamScheduler():
    """Updates KFAC hyper-parameters at each epoch
    Args:
      kfac (KFAC): wrapped KFAC preconditioner
      damping_alpha (float): multiplicative factor of the damping (default: 1)
      damping_schedule (list): list of epochs to multiply the damping by `damping_alpha` (default: None)
      update_freq_alpha (float): multiplicative factor of the KFAC update freq (default: 1)
      update_freq_schedule (list): list of epochs to multiply the KFAC update freq by `update_freq_alpha` (default: None)
      start_epoch (int): starting epoch, for use if resuming training from checkpoint (default: 0)
    """
    def __init__(self,
                 kfac,
                 damping_alpha=1,
                 damping_schedule=None,
                 update_freq_alpha=1,
                 update_freq_schedule=None,
                 start_epoch=0):

        self.kfac = kfac
        params = self.kfac.param_groups[0]

        self.damping_base = params['damping']
        self.damping_alpha = damping_alpha
        self.damping_schedule = damping_schedule
        self.damping_factor_func = \
                self._get_factor_func(self.damping_schedule,
                                     self.damping_alpha)

        self.fac_update_freq_base = params['fac_update_freq']
        self.kfac_update_freq_base = params['kfac_update_freq']
        self.update_freq_alpha = update_freq_alpha
        self.update_freq_schedule = update_freq_schedule
        self.update_freq_factor_func = \
                self._get_factor_func(self.update_freq_schedule,
                                     self.update_freq_alpha)

        self.epoch = start_epoch

    def _get_factor_func(self, schedule, alpha):
        """Returns a function to compute an update factor using the epoch"""
        if schedule is not None:
            schedule.sort(reverse=True)
        else:
            schedule = []

        def factor_func(epoch):
            factor = 1.
            for e in schedule:
                if epoch >= e:
                    factor *= alpha
            return factor

        return factor_func

    def step(self, epoch=None):
        """Update KFAC parameters"""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        params = self.kfac.param_groups[0]

        params['damping'] = self.damping_base * self.damping_factor_func(self.epoch)

        factor = self.update_freq_factor_func(self.epoch)
        params['fac_update_freq'] = int(self.fac_update_freq_base * factor)
        params['kfac_update_freq'] = int(self.kfac_update_freq_base * factor)

