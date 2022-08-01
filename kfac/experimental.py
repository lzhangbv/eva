import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
import kfac.backend as backend

from kfac.utils import get_vector_a, get_vector_g
import logging
logger = logging.getLogger()

"""
Eva Implemetation for Experiments, e.g., different clip methods
"""

class KFAC(optim.Optimizer):
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
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
                 kl_clip=0.01,
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

        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip > 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0

    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_a(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_a:
                    self.m_a[module] = new
                else:
                    #self.m_a[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_a[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
            if backend.comm.size() > 1:
                self.handles.append(backend.comm.allreduce_async_(self.m_a[module], op=backend.comm.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_g:
                    self.m_g[module] = new
                else:
                    #self.m_g[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_g[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
            if backend.comm.size() > 1:
                self.handles.append(backend.comm.allreduce_async_(self.m_g[module], op=backend.comm.Average))

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

	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via Eva"""
        vg_sum = 0
        g_sum = 0
        v_sum = 0
        # other clip methods (WIP...)
        self.clip_type = 'kl-clip'  # choices: kl-clip, grad-clip, grad-scale, lars-clip, rms-clip

        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1)
            mg = self.m_g[module].view(-1, 1)
            grad = self._get_grad(module)
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            ag = (mg.T @ grad @ ma).item()

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))
            v.add_(grad)
            v.div_(self.damping)

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # kl clip: grad and precon grad
                if self.kl_clip is not None:
                    if self.clip_type == 'kl-clip':  # without-lr
                        vg_sum += (weight * module.weight.grad.data).sum().item()
                        vg_sum += (bias * module.bias.grad.data).sum().item()
                    elif self.clip_type == 'grad-clip':
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                    elif self.clip_type == 'grad-scale':
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                        g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    if self.clip_type == 'kl-clip':  # without-lr
                        vg_sum += (weight * module.weight.grad.data).sum().item()
                    elif self.clip_type == 'grad-clip':
                        v_sum += (weight * weight).sum().item()
                    elif self.clip_type == 'grad-scale':
                        v_sum += (weight * weight).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
            del v

            # lars-clip or rms-clip
            if self.kl_clip is not None:
                if self.clip_type == 'lars-clip':
                    w_norm = module.weight.norm(2).item()
                    v_norm = module.weight.grad.norm(2).item()
                    if v_norm > 0:
                        nu = min(50, self.kl_clip * w_norm / v_norm)
                        module.weight.grad.data.mul_(nu)
                    if module.bias is not None:
                        w_norm = module.bias.norm(2).item()
                        v_norm = module.bias.grad.norm(2).item()
                        if v_norm > 0:
                            nu = min(50, self.kl_clip * w_norm / v_norm)
                            module.bias.grad.data.mul_(nu)
                elif self.clip_type == 'rms-clip':
                    rms = module.weight.grad.norm(2) / (module.weight.grad.numel() ** 0.5)
                    module.weight.grad.data.div_((rms / self.rms_clip).clamp_(min=1.0))
                    if module.bias is not None:
                        rms = module.bias.grad.norm(2) / (module.bias.grad.numel() ** 0.5)
                        module.bias.grad.data.div_((rms / self.rms_clip).clamp_(min=1.0))

        # kl-clip or grad-clip or grad-scale
        if self.kl_clip is not None:
            if self.clip_type == 'kl-clip':
                nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))
            elif self.clip_type == 'grad-clip':
                nu = min(1.0, math.sqrt(self.grad_clip / abs(v_sum)))
            elif self.clip_type == 'grad-scale':
                nu = math.sqrt(g_sum / v_sum)

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

        if self.steps % self.fac_update_freq == 0 and backend.comm.size() > 1:
            for handle in self.handles:
                backend.comm.synchronize(handle)
            self.handles = []
        
        self._precondition_grads()

        self.steps += 1
