# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of Shampoo."""

from __future__ import print_function

import enum
import itertools

from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim


@dataclass
class ShampooHyperParams:
  """Shampoo hyper parameters."""
  damping: float = 0.03
  beta2: float = 1.0
  weight_decay: float = 0.0
  # How often to compute statistics.
  statistics_compute_steps: int = 1
  # Nesterov momentum
  nesterov: bool = True


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, var, hps):
    self._hps = hps
    self._shape = var.shape

    rank = len(self._shape)
    device = var.get_device()
    if rank <= 1: # skip vectors
        self.statistics = []
    else:
        #self.statistics = [1e-12 * torch.ones(s, device=device) for s in self._shape]
        self.statistics = [torch.zeros(s, device=device) for s in self._shape]

  def add_statistics(self, grad):
    """Compute statistics from gradients and add to the correct state entries.

    Args:
      grad: Gradient to compute statistics from.
    """
    if not self.statistics: return

    w1 = self._hps.beta2
    w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
    rank = len(self._shape)
    for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = torch.mean(grad, dim=axes)
        self.statistics[i].mul_(w1).add_(stat, alpha=w2)


  def preconditioned_grad(self, grad, damping):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.

    Returns:
      A preconditioned gradient.
    """
    if not self.statistics: return grad

    rank = len(grad.shape)
    precond_grad = grad.clone().detach()
    mu = 1
    for i in range(rank):
        v = self.statistics[i].view(-1, 1)
        precond_grad = torch.tensordot(precond_grad, v @ v.T, [[0], [0]])
        mu *= torch.norm(v) ** 2

    precond_grad.mul_(-1/(damping+mu)).add_(grad)
    #precond_grad.div_(damping)
    
    # Layerwise scale: scale preconditioned grad to grad's magnitude
    grad_norm = torch.norm(grad)
    precond_norm = torch.norm(precond_grad)
    precond_grad.mul_(grad_norm / (precond_norm + 1e-16))

    return precond_grad


STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'


class vShampoo(optim.Optimizer):
  """The Shampoo optimizer."""

  def __init__(self,
               params,
               lr=1.0,
               momentum=0.9,
               weight_decay=0.0, 
               statistics_compute_steps=1, 
               preconditioning_compute_steps=1, 
               damping=0.03, 
               beta2=1.0,
               hyperparams=ShampooHyperParams()):
    defaults = dict(lr=lr, momentum=momentum)
    self.hps = hyperparams
    self.hps.weight_decay = weight_decay
    self.hps.statistics_compute_steps = statistics_compute_steps
    self.hps.preconditioning_compute_steps = preconditioning_compute_steps
    self.hps.damping = damping
    self.hps.beta2 = beta2
    super(vShampoo, self).__init__(params, defaults)

  def init_var_state(self, var, state):
    """Initialize the PyTorch state of for a single variable."""
    state[STEP] = 0
    state[MOMENTUM] = torch.zeros_like(var.data, device=var.get_device())
    state[PRECONDITIONER] = Preconditioner(var, self.hps)

  def step(self, closure=None):
    hps = self.hps
    for group in self.param_groups:
      lr = group['lr']
      for p in group['params']:
        if p.grad is None: continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Shampoo does not support sparse yet')
        state = self.state[p]
        if not state:
          self.init_var_state(p, state)

        preconditioner = state[PRECONDITIONER]

        # Gather statistics, compute preconditioners
        if state[STEP] % hps.statistics_compute_steps == 0:
          preconditioner.add_statistics(grad)
        
        state[STEP] += 1

        # Precondition gradients
        shampoo_grad = preconditioner.preconditioned_grad(grad, self.hps.damping)

        # Weight decay
        if self.hps.weight_decay != 0.0:
          shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)

        # Momentum and Nesterov momentum, if needed
        state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)

        momentum_update = state[MOMENTUM]
        wd_update = shampoo_grad

        if hps.nesterov:
          momentum_update.mul_(group['momentum']).add_(wd_update)

        # Final update
        p.data.add_(momentum_update, alpha=-lr)
        
