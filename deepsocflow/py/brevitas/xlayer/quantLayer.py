import torch.nn as nn

from brevitas.nn.quant_linear import QuantLinear as _QuantLinear
from brevitas.nn.quant_conv import QuantConv1d as _QuantConv1d
from brevitas.nn.quant_conv import QuantConv2d as _QuantConv2d
from brevitas.nn.quant_conv import QuantConv3d as _QuantConv3d
from brevitas.nn.quant_rnn import QuantRNN as _QuantRNN
from brevitas.nn.quant_rnn import QuantLSTM as _QuantLSTM

class QuantLinear(_QuantLinear):
		# Arguments:
		#   - in_features (int): size of each input sample
		#   - out_features (int): size of each output sample
		#   - bias (bool, optional): if True, adds a learnable bias to the output. Default: True
		#   - weight_bit_width (int, optional): bit width for weight quantization. Default: 8
		#   - input_quant (optional): activation quantizer applied to the input. Default: None (unquantized)
		#   - input_bit_width (int, optional): bit width for input_quant (requires input_quant to be set)
		#   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
		#
		# Input:  x, shape (N, in_features) - Tensor or QuantTensor
		# Output: shape (N, out_features) - Tensor, or IntQuantTensor if return_quant_tensor=True
		#
		# Usage:
		#   lin = QuantLinear(16, 4, bias=True, weight_bit_width=4)
		#   y = lin(x)  # 4-bit weights
		pass

class QuantConv1d(_QuantConv1d):
		# Arguments: same as QuantConv2d (in_channels, out_channels, kernel_size, stride,
		#   padding, dilation, groups, bias, weight_bit_width, input_quant, input_bit_width,
		#   return_quant_tensor), applied along a single spatial dimension.
		#
		# Input:  x, shape (N, in_channels, L) - Tensor or QuantTensor
		# Output: shape (N, out_channels, L_out) - Tensor, or IntQuantTensor if return_quant_tensor=True
		#
		# Usage:
		#   conv = QuantConv1d(3, 8, kernel_size=3, padding=1, bias=False)
		#   y = conv(x)  # 8-bit weights (default)
		pass

class QuantConv2d(_QuantConv2d):
		# Arguments:
		#   - in_channels (int): number of channels in the input
		#   - out_channels (int): number of channels produced by the convolution
		#   - kernel_size (int or tuple): size of the convolving kernel
		#   - stride (int or tuple, optional): stride of the convolution. Default: 1
		#   - padding (int, tuple or str, optional): padding added to all sides of the input. Default: 0
		#   - dilation (int or tuple, optional): spacing between kernel elements. Default: 1
		#   - groups (int, optional): number of blocked connections input -> output channels. Default: 1
		#   - bias (bool, optional): if True, adds a learnable bias to the output. Default: True
		#   - weight_bit_width (int, optional): bit width for weight quantization. Default: 8
		#   - input_quant (optional): activation quantizer applied to the input. Default: None (unquantized)
		#   - input_bit_width (int, optional): bit width for input_quant (requires input_quant to be set)
		#   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
		#
		# Input:  x, shape (N, in_channels, H, W) - Tensor or QuantTensor
		# Output: shape (N, out_channels, H_out, W_out) - Tensor, or IntQuantTensor if return_quant_tensor=True
		#
		# Usage:
		#   conv = QuantConv2d(3, 8, kernel_size=3, padding=1, bias=True)
		#   y = conv(x)  # 8-bit weights, unquantized activations
		#
		#   conv4 = QuantConv2d(3, 8, kernel_size=3, padding=1, bias=False, weight_bit_width=4)
		#   y = conv4(x)  # 4-bit weights
		pass

class QuantConv3d(_QuantConv3d):
		# Arguments: same as QuantConv2d (in_channels, out_channels, kernel_size, stride,
		#   padding, dilation, groups, bias, weight_bit_width, input_quant, input_bit_width,
		#   return_quant_tensor), applied over three spatial dimensions.
		#
		# Input:  x, shape (N, in_channels, D, H, W) - Tensor or QuantTensor
		# Output: shape (N, out_channels, D_out, H_out, W_out) - Tensor, or IntQuantTensor if return_quant_tensor=True
		#
		# Usage:
		#   conv = QuantConv3d(3, 8, kernel_size=3, padding=1, bias=False)
		#   y = conv(x)  # 8-bit weights (default)
		pass

class QuantRNN(_QuantRNN):
		# Arguments:
		#   - input_size (int): number of expected features in the input
		#   - hidden_size (int): number of features in the hidden state
		#   - num_layers (int, optional): number of stacked recurrent layers. Default: 1
		#   - nonlinearity (str, optional): 'tanh' or 'relu'. Default: 'tanh'
		#   - bias (bool, optional): if True, adds learnable biases. Default: True
		#   - batch_first (bool, optional): if True, input/output are (batch, seq, feature). Default: False
		#   - bidirectional (bool, optional): if True, becomes a bidirectional RNN. Default: False
		#   - weight_quant/bias_quant/io_quant/gate_acc_quant (optional): quantizers for
		#       weights (default 8-bit), bias (default 32-bit), input/output activations and
		#       gate accumulators (default 8-bit) respectively
		#   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
		#
		# Input:  x, shape (seq_len, N, input_size) by default (batch_first=False)
		# Output: (out, h_n)
		#   - out: shape (seq_len, N, hidden_size)
		#   - h_n: shape (num_layers, N, hidden_size)
		#
		# Usage:
		#   rnn = QuantRNN(input_size=10, hidden_size=20, num_layers=1)
		#   out, h_n = rnn(x)
		pass

class QuantLSTM(_QuantLSTM):
		# Arguments: same as QuantRNN (input_size, hidden_size, num_layers, bias,
		#   batch_first, bidirectional, weight_quant, bias_quant, io_quant,
		#   gate_acc_quant, return_quant_tensor), plus LSTM-specific quantizers
		#   (sigmoid_quant, tanh_quant, cell_state_quant, default 8-bit).
		#
		# Input:  x, shape (seq_len, N, input_size) by default (batch_first=False)
		# Output: (out, (h_n, c_n))
		#   - out: shape (seq_len, N, hidden_size)
		#   - h_n, c_n: shape (num_layers, N, hidden_size)
		#
		# Usage:
		#   lstm = QuantLSTM(input_size=10, hidden_size=20, num_layers=1)
		#   out, (h_n, c_n) = lstm(x)
		pass


class QuantConv1dBN(nn.Module):
	# QuantConv1d fused with BatchNorm1d, with a settable .act attribute so it plugs
	# directly into XBundle (which calls core(x) then core.act(x)).
	#
	# Arguments:
	#   - in_channels, out_channels, kernel_size, stride, padding: same as QuantConv1d.
	#       Convolution always has bias=False (BatchNorm makes a conv bias redundant).
	#   - act (nn.Module, optional): activation applied after conv+BN by the caller
	#       (e.g. XBundle), not inside forward(). Default: None.
	#   - **kwargs: forwarded to QuantConv1d (e.g. weight_bit_width, input_quant, ...)
	#
	# Input:  x, shape (N, in_channels, L)
	# Output: shape (N, out_channels, L_out) - BatchNorm'd conv output (no act applied)
	#
	# Usage:
	#   core = QuantConv1dBN(3, 8, kernel_size=3, stride=1, padding=1, act=QuantReLU())
	#   y = core(x)       # conv -> batchnorm
	#   y = core.act(y)   # activation applied separately (this is what XBundle does)
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=None, **kwargs):
		super().__init__()
		self.conv = QuantConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, **kwargs)
		self.bn = nn.BatchNorm1d(out_channels)
		self.act = act

	def forward(self, x):
		return self.bn(self.conv(x))


class QuantConv2dBN(nn.Module):
	# QuantConv2d fused with BatchNorm2d, with a settable .act attribute so it plugs
	# directly into XBundle (which calls core(x) then core.act(x)).
	#
	# Arguments:
	#   - in_channels, out_channels, kernel_size, stride, padding: same as QuantConv2d.
	#       Convolution always has bias=False (BatchNorm makes a conv bias redundant).
	#   - act (nn.Module, optional): activation applied after conv+BN by the caller
	#       (e.g. XBundle), not inside forward(). Default: None.
	#   - **kwargs: forwarded to QuantConv2d (e.g. weight_bit_width, input_quant, ...)
	#
	# Input:  x, shape (N, in_channels, H, W)
	# Output: shape (N, out_channels, H_out, W_out) - BatchNorm'd conv output (no act applied)
	#
	# Usage:
	#   core = QuantConv2dBN(64, 64, kernel_size=3, stride=1, padding=1, act=QuantReLU())
	#   y = core(x)       # conv -> batchnorm
	#   y = core.act(y)   # activation applied separately (this is what XBundle does)
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=None, **kwargs):
		super().__init__()
		self.conv = QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels)
		self.act = act

	def forward(self, x):
		return self.bn(self.conv(x))


class QuantConv3dBN(nn.Module):
	# QuantConv3d fused with BatchNorm3d, with a settable .act attribute so it plugs
	# directly into XBundle (which calls core(x) then core.act(x)).
	#
	# Arguments:
	#   - in_channels, out_channels, kernel_size, stride, padding: same as QuantConv3d.
	#       Convolution always has bias=False (BatchNorm makes a conv bias redundant).
	#   - act (nn.Module, optional): activation applied after conv+BN by the caller
	#       (e.g. XBundle), not inside forward(). Default: None.
	#   - **kwargs: forwarded to QuantConv3d (e.g. weight_bit_width, input_quant, ...)
	#
	# Input:  x, shape (N, in_channels, D, H, W)
	# Output: shape (N, out_channels, D_out, H_out, W_out) - BatchNorm'd conv output (no act applied)
	#
	# Usage:
	#   core = QuantConv3dBN(3, 8, kernel_size=3, stride=1, padding=1, act=QuantReLU())
	#   y = core(x)       # conv -> batchnorm
	#   y = core.act(y)   # activation applied separately (this is what XBundle does)
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=None, **kwargs):
		super().__init__()
		self.conv = QuantConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, **kwargs)
		self.bn = nn.BatchNorm3d(out_channels)
		self.act = act

	def forward(self, x):
		return self.bn(self.conv(x))

