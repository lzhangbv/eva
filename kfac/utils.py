import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x

def get_vector_a(a, layer):
    """Return vectorized input activation (m_a)"""
    if isinstance(layer, nn.Linear): 
        a = torch.mean(a, list(range(len(a.shape)))[0:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a

    elif isinstance(layer, nn.Conv2d):
        # batch averag first
        a = torch.mean(a, dim=0, keepdim=True)
        # extract patch
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a = torch.mean(a, [0, 1, 2])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a
        
    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_vector_g(g, layer):
    """Return vectorized deviation w.r.t. the pre-activation output (m_g)"""
    if isinstance(layer, nn.Linear):
        g = torch.mean(g, list(range(len(g.shape)))[0:-1])
        return g

    elif isinstance(layer, nn.Conv2d):
        g = torch.mean(g, [0, 2, 3])
        return g

    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_factor_A(a, layer):
    """Return KF A"""
    if isinstance(layer, nn.Linear): 
        if len(a.shape) > 2:
            a = torch.mean(a, list(range(len(a.shape)))[1:-1]) # average on sequential dim (if any)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / a.size(0))

    elif isinstance(layer, nn.Conv2d):
        # extract patch
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a = torch.mean(a, [1, 2])  # average on spatial dims
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / a.size(0))
        
    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_factor_G(g, layer):
    """Return KF G"""
    if isinstance(layer, nn.Linear):
        if len(g.shape) > 2:
            g = torch.mean(g, list(range(len(g.shape)))[1:-1]) # average on sequential dim (if any)
        return g.t() @ (g / g.size(0))

    elif isinstance(layer, nn.Conv2d):
        g = torch.mean(g, [2, 3]) # average on spatial dims
        return g.t() @ (g / g.size(0))

    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def mat_inv(x):
    u = torch.linalg.cholesky(x)
    return torch.cholesky_inverse(u)
