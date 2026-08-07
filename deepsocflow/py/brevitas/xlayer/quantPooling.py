import torch.nn as nn

from brevitas.nn.quant_avg_pool import TruncAvgPool2d as _TruncAvgPool2d
from brevitas.nn.quant_avg_pool import TruncAdaptiveAvgPool2d as _TruncAdaptiveAvgPool2d

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

class QuantMaxPool1d(nn.MaxPool1d):
		# Plain (unquantized) max pool, 1D. See QuantMaxPool2d below for why max pooling
		# needs no quantization arguments.
		#
		# Arguments: same as torch.nn.MaxPool1d (kernel_size, stride, padding, dilation,
		#   return_indices, ceil_mode).
		#
		# Input:  x, shape (N, C, L) - Tensor or QuantTensor
		# Output: shape (N, C, L_out) - same type as input
		#
		# Usage:
		#   pool = QuantMaxPool1d(kernel_size=3, stride=2, padding=1)
		#   y = pool(x)
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

class QuantMaxPool3d(nn.MaxPool3d):
		# Plain (unquantized) max pool, 3D. See QuantMaxPool2d above for why max pooling
		# needs no quantization arguments.
		#
		# Arguments: same as torch.nn.MaxPool3d (kernel_size, stride, padding, dilation,
		#   return_indices, ceil_mode).
		#
		# Input:  x, shape (N, C, D, H, W) - Tensor or QuantTensor
		# Output: shape (N, C, D_out, H_out, W_out) - same type as input
		#
		# Usage:
		#   pool = QuantMaxPool3d(kernel_size=3, stride=2, padding=1)
		#   y = pool(x)
		pass

class QuantAdaptiveMaxPool1d(nn.AdaptiveMaxPool1d):
		# Adaptive (output-size-driven) version of QuantMaxPool1d.
		#
		# Arguments:
		#   - output_size (int or tuple): target output length, e.g. 1
		#   - return_indices (bool, optional): also return the indices of the max values. Default: False
		#
		# Input:  x, shape (N, C, L) - Tensor or QuantTensor
		# Output: shape (N, C, output_size) - same type as input
		#
		# Usage:
		#   pool = QuantAdaptiveMaxPool1d(output_size=1)
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

class QuantAdaptiveMaxPool3d(nn.AdaptiveMaxPool3d):
		# Adaptive (output-size-driven) version of QuantMaxPool3d.
		#
		# Arguments:
		#   - output_size (int or tuple): target output spatial size, e.g. (1, 1, 1)
		#   - return_indices (bool, optional): also return the indices of the max values. Default: False
		#
		# Input:  x, shape (N, C, D, H, W) - Tensor or QuantTensor
		# Output: shape (N, C, *output_size) - same type as input
		#
		# Usage:
		#   pool = QuantAdaptiveMaxPool3d(output_size=(1, 1, 1))
		#   y = pool(x)
		pass
