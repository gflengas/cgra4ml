from brevitas.nn.quant_linear import QuantLinear as _QuantLinear
from brevitas.nn.quant_conv import QuantConv1d as _QuantConv1d
from brevitas.nn.quant_conv import QuantConv2d as _QuantConv2d
from brevitas.nn.quant_conv import QuantConv3d as _QuantConv3d
from brevitas.nn.quant_rnn import QuantRNN as _QuantRNN
from brevitas.nn.quant_rnn import QuantLSTM as _QuantLSTM

class QuantLinear(_QuantLinear):
		pass

class QuantConv1d(_QuantConv1d):
		pass

class QuantConv2d(_QuantConv2d):
		pass

class QuantConv3d(_QuantConv3d):
		pass

