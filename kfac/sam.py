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
    """Distributed SAM optimizer that approximates sam-gradient with Kronecker factorization. 
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
                 damping=0.03, # not used
                 fac_update_freq=1,  
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 kl_clip=0.01,  # refer to neighborhood_size
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
        self.kfac_update_freq = kfac_update_freq    # freq for communicating KFs
        self.kfac_batch_size = kfac_batch_size      # subsampled batch size for computing KFs
        self.neighborhood_size = kl_clip if (kl_clip is not None and kl_clip > 0) else 0.01
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_A, self.m_G = {}, {}
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

	### Sharpness-aware minimization gradients
    def _compute_sam_grads(self):
        """Compute sam gradients via K-FAC"""
        # accumulating g_sum
        g_sum = 0
        for module in self.modules:
            g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
            if module.bias is not None:
                g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()
        g_sum = math.sqrt(g_sum)
        #if backend.comm.rank() == 0:
        #    logger.info("sqrt g_sum: %f; alpha: %f" % (g_sum, self.neighborhood_size/g_sum))
        
        for module in self.modules:
            # compute sam grads
            grad = self._get_grad(module)
            v = self.m_G[module] @ grad @ self.m_A[module]

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                module.weight.grad.data.add_(weight, alpha=self.neighborhood_size/g_sum)
                module.bias.grad.data.add_(bias, alpha=self.neighborhood_size/g_sum)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                module.weight.grad.data.add_(weight, alpha=self.neighborhood_size/g_sum)
            del v

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
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        if self.steps % self.kfac_update_freq == 0:
            if backend.comm.size() > 1:
                for handle in self.handles:
                    backend.comm.synchronize(handle)
                self.handles = []
        
        self._compute_sam_grads()
        self.steps += 1
