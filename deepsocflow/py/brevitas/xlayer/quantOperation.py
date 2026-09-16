import torch
import torch.nn as nn


class QuantResidualAdd(nn.Module):
    # Elementwise add of a bundle's output with a residual/skip-connection source,
    # followed by an activation. Used by XBundle when add_act is set (xbundle.py:36-39).
    #
    # Arguments:
    #   - act (nn.Module): activation applied after the sum. Required - raises ValueError
    #       if None; pass an identity activation (e.g. nn.Identity()) if no nonlinearity
    #       is needed, instead of leaving this unset.
    #   - sys_bits (optional): reserved for hardware bit-width config. Default: None
    #
    # Input:  x, x_add - two Tensors of the same shape
    # Output: act(x + x_add) - Tensor of the same shape
    #
    # Usage:
    #   add = QuantResidualAdd(act=QuantReLU())
    #   y = add(x, x_skip)
    def __init__(self, act, sys_bits=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if act is None:
            raise ValueError("Activation function must be provided. Set type to none if no activation is needed")
        self.act = act
        self.sys_bits = sys_bits
        self.source_ib = None  # bundle index (ib) of the residual/skip-connection source

    def forward(self, x, x_add):
        return self.act(x + x_add)

    def call_int(self, x, hw):
        raise NotImplementedError(
            "QuantResidualAdd.call_int requires an XTensor port (add_val_shift) for the brevitas backend"
        )


class QuantTranspose(nn.Module):
    # Swaps two dimensions of a tensor (e.g. NCHW <-> NHWC layout changes needed when
    # bridging between hardware and framework tensor layouts).
    #
    # Arguments:
    #   - dim0 (int): first dimension to swap
    #   - dim1 (int): second dimension to swap
    #
    # Input:  x, any shape
    # Output: x with dim0 and dim1 swapped
    #
    # Usage:
    #   t = QuantTranspose(1, 3)      # NCHW -> NWHC
    #   y = t(x)
    def __init__(self, dim0, dim1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

    def call_int(self, x, hw):
        raise NotImplementedError(
            "QuantTranspose.call_int requires an XTensor port for the brevitas backend"
        )


class QuantConcatenate(nn.Module):
    # Concatenates multiple bundle outputs along a dimension (e.g. DenseNet-style
    # feature concatenation, as opposed to QuantResidualAdd's elementwise sum).
    #
    # Arguments:
    #   - dim (int, optional): dimension to concatenate along. Default: 1 (channel dim, NCHW)
    #
    # Input:  xs - a list/tuple of Tensors, matching in every dimension except dim
    # Output: Tensor, xs concatenated along dim
    #
    # Usage:
    #   cat = QuantConcatenate(dim=1)
    #   y = cat([x1, x2])
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)

    def call_int(self, xs, hw):
        raise NotImplementedError(
            "QuantConcatenate.call_int requires an XTensor port for the brevitas backend"
        )
