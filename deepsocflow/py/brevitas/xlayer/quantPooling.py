import torch
import torch.nn as nn
import torch.nn.functional as F

from brevitas.nn.quant_avg_pool import TruncAvgPool2d as _TruncAvgPool2d
from brevitas.nn.quant_avg_pool import TruncAdaptiveAvgPool2d as _TruncAdaptiveAvgPool2d
from brevitas.quant_tensor import IntQuantTensor

class QuantAvgPool2d(_TruncAvgPool2d):
		# Average pool that truncates the accumulated sum to a fixed bit width, matching
		# how a hardware accumulator would round after averaging (bit-accurate pooling).
		#
		# Arguments:
		#   - kernel_size (int or tuple): size of the pooling window
		#   - stride (int or tuple, optional): stride of the window. Default: kernel_size
		#   - ceil_mode (bool, optional): use ceil instead of floor for output shape. Default: False
		#   - count_include_pad (bool, optional): include zero-padding in the average. Default: True
		#   - divisor_override (int, optional): use this value as the divisor instead of the pool size
		#   - trunc_quant (optional): truncation quantizer. Default: RoundTo8bit (8-bit)
		#   - bit_width (int, optional): overrides trunc_quant's bit width, e.g. 4
		#   - return_quant_tensor (bool, optional): return an IntQuantTensor. Default: True
		#
		# Input:  x - QuantTensor (NOT a plain Tensor - raises AssertionError otherwise,
		#     since truncation needs the input's existing quantization scale). Feed it the
		#     output of an upstream layer with return_quant_tensor=True, e.g. QuantReLU(...).
		# Output: shape (N, C, H_out, W_out) - IntQuantTensor
		#
		# Usage:
		#   relu = QuantReLU(return_quant_tensor=True)
		#   pool = QuantAvgPool2d(kernel_size=2, stride=2, bit_width=4)
		#   y = pool(relu(x))
		pass

class QuantAdaptiveAvgPool2d(_TruncAdaptiveAvgPool2d):
		# Adaptive (output-size-driven) version of QuantAvgPool2d - same truncation
		# behavior and the same QuantTensor input requirement, but the window/stride are
		# computed from the target output_size instead of being fixed, so it works for
		# any input resolution (e.g. a global-average-pool classifier head).
		#
		# Arguments:
		#   - output_size (int or tuple): target output spatial size, e.g. (1, 1)
		#   - trunc_quant (optional): truncation quantizer. Default: RoundTo8bit (8-bit)
		#   - bit_width (int, optional): overrides trunc_quant's bit width, e.g. 4
		#   - return_quant_tensor (bool, optional): return an IntQuantTensor. Default: True
		#
		# Input:  x - QuantTensor (NOT a plain Tensor - see QuantAvgPool2d above)
		# Output: shape (N, C, *output_size) - IntQuantTensor
		#
		# Usage:
		#   relu = QuantReLU(return_quant_tensor=True)
		#   pool = QuantAdaptiveAvgPool2d(output_size=(1, 1))
		#   y = pool(relu(x))  # global average pool, any input H/W
		pass

class QuantMaxPool2d(nn.MaxPool2d):
		# Plain (unquantized) max pool - max selects an existing value rather than
		# accumulating one, so there is nothing to truncate/requantize. Works on a plain
		# Tensor or a QuantTensor (passes it through unchanged, picking the max element).
		#
		# Arguments: same as torch.nn.MaxPool2d (kernel_size, stride, padding, dilation,
		#   return_indices, ceil_mode) - no quantization arguments.
		#
		# Input:  x, shape (N, C, H, W) - Tensor or QuantTensor
		# Output: shape (N, C, H_out, W_out) - same type as input
		#
		# Usage:
		#   pool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
		#   y = pool(x)
		pass

class QuantAdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
		# Adaptive (output-size-driven) version of QuantMaxPool2d - same unquantized
		# passthrough behavior, but the window/stride are computed from output_size.
		#
		# Arguments:
		#   - output_size (int or tuple): target output spatial size, e.g. (1, 1)
		#   - return_indices (bool, optional): also return the indices of the max values. Default: False
		#
		# Input:  x, shape (N, C, H, W) - Tensor or QuantTensor
		# Output: shape (N, C, *output_size) - same type as input
		#
		# Usage:
		#   pool = QuantAdaptiveMaxPool2d(output_size=(1, 1))
		#   y = pool(x)
		pass


def _c_div(a, b):
	"""C integer division on a float tensor holding integers: truncates toward
	zero, where torch's floor_divide floors."""
	q = torch.trunc(torch.abs(a) / abs(b))
	return torch.where(a < 0, -q, q)


def div_round_torch(a, b):
	"""runtime.h's div_round, on a tensor of integer-valued floats.

	Mirrors deepsocflow/py/brevitas/simulation/sim.py::div_round exactly; that one is pinned
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

	The hardware sums a pooling window in integers and then divides with
	`runtime.h`'s `div_round` macro. That macro is not round-half-away-from-zero: it
	deviates by up to 1 LSB on roughly half of all inputs, almost always on negative
	ones (`div_round(-4, 4)` is `0`, not `-1`). A float average, or QuantAvgPool2d
	above (brevitas's TruncAvgPool2d, which sums and then *truncates to a bit
	width* - a shift rather than a divide), both disagree with it.

	So this module computes what the hardware computes. That is the same choice the
	LUT work landed on: a model whose accuracy figure does not describe what the
	hardware will do is worse than a slower one that does. The cost is that average
	pooling here is genuinely a coarser operation than `nn.AvgPool2d`; calibration
	and QAT can at least see that, which they cannot if the divergence is hidden
	until deployment.

	Input and output share a scale: the average of values on a grid stays on that
	grid, so nothing is requantized and the output is a valid QuantTensor with the
	incoming scale, zero point and bit width.

	Only 'valid' padding (torch `padding=0`) is implemented. Under 'same' the
	engine's divisor shrinks at the borders (`count` comes from the clipped window,
	`runtime.h:535`), which is a separate and larger piece of work.
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
