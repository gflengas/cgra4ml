"""Average pooling that reproduces the engine's arithmetic exactly.

The hardware sums a pooling window in integers and then divides with
`runtime.h`'s `div_round` macro. That macro is not round-half-away-from-zero: it
deviates by up to 1 LSB on roughly half of all inputs, almost always on negative
ones (`div_round(-4, 4)` is `0`, not `-1`). A float average, or brevitas's own
`TruncAvgPool2d` - which sums and then *truncates to a bit width*, a shift rather
than a divide - both disagree with it.

So this module computes what the hardware computes. That is the same choice the
LUT work landed on: a model whose accuracy figure does not describe what the
hardware will do is worse than a slower one that does. The cost is that average
pooling here is genuinely a coarser operation than `nn.AvgPool2d`; calibration
and QAT can at least see that, which they cannot if the divergence is hidden
until deployment.

Only 'valid' padding (torch `padding=0`) is implemented. Under 'same' the
engine's divisor shrinks at the borders (`count` comes from the clipped window,
`runtime.h:535`), which is a separate and larger piece of work.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.quant_tensor import IntQuantTensor


def _c_div(a, b):
    """C integer division on a float tensor holding integers: truncates toward
    zero, where torch's floor_divide floors."""
    q = torch.trunc(torch.abs(a) / abs(b))
    return torch.where(a < 0, -q, q)


def div_round_torch(a, b):
    """runtime.h's div_round, on a tensor of integer-valued floats.

    Mirrors deepsocflow/py/brevitas/sim.py::div_round exactly; that one is pinned
    against the compiled C macro, and this one is pinned against it in turn.
    """
    correction = torch.bitwise_and(
        torch.bitwise_not(torch.bitwise_or(
            torch.full_like(a, b, dtype=torch.int64),
            _c_div(a, b).to(torch.int64))),
        1).to(a.dtype)
    return _c_div(a + (b // 2) - correction, b)


class QuantAvgPool2dDivRound(nn.Module):
    """Average pool over a QuantTensor using the engine's integer arithmetic.

    Input and output share a scale: the average of values on a grid stays on that
    grid, so nothing is requantized and the output is a valid QuantTensor with the
    incoming scale, zero point and bit width.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 2
        self.count = int(kh) * int(kw)
        assert (padding == 0 or padding == (0, 0)), (
            f"pool padding {padding} is not supported - only 'valid' pooling is "
            f"implemented, because the engine's divisor shrinks at the borders "
            f"under 'same' padding")

    def forward(self, x):
        if not isinstance(x, IntQuantTensor):
            # brevitas's calibration_mode disables the upstream activation's
            # quantizer while it collects statistics, so the pool is handed a plain
            # tensor on that pass. There is no grid to work on yet and the values
            # only feed scale statistics, so an ordinary mean is both the best
            # available answer and harmless. Every real inference pass takes the
            # integer path below.
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

        # avg_pool2d gives the mean; multiplying by the (constant, 'valid') window
        # size recovers the integer sum the engine accumulates. The round() is not
        # a rounding decision, just undoing float division error on an exact value.
        ints = torch.round(x.value / x.scale)
        window_sum = torch.round(
            F.avg_pool2d(ints, self.kernel_size, self.stride, self.padding) * self.count)
        out_ints = div_round_torch(window_sum, self.count)

        return IntQuantTensor(
            out_ints * x.scale, x.scale, x.zero_point, x.bit_width, x.signed, x.training)
