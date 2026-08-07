import torch
import torch.nn as nn
from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_qonnx

from deepsocflow.py.brevitas.utils import reset_bundles
from deepsocflow.py.brevitas.xbundle.xbundle import XBundle
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d
from deepsocflow.py.brevitas.xlayer.quantActivation import (
	QuantIdentity, QuantReLU, QuantSigmoid, QuantTanh,
	QuantLeakyReLU, QuantSiLU, QuantSELU, QuantGELU,
)

# model json or model.py

# torch -> high precision -> quantization -> export to qonnx
# or brevitas -> low precision -> export to qonnx

# qonnx json

# cgra4ml

# Compute layers: plain torch type -> our xlayer quant equivalent.
LAYER_MAP = {
	nn.Linear: QuantLinear,
	nn.Conv1d: QuantConv1d,
	nn.Conv2d: QuantConv2d,
	nn.Conv3d: QuantConv3d,
	# layers with batchnorm -> conv2dBN, linearBN, etc.
}

# Activations: plain torch type -> our xlayer quant equivalent. Covers LeakyReLU/SiLU/
# SELU/GELU, which brevitas.nn doesn't ship built-in (see xlayer/quantActivation.py) -
# the reason this maps through our own xlayer instead of brevitas.graph.quantize.
ACT_MAP = {
	nn.ReLU: QuantReLU,
	nn.Sigmoid: QuantSigmoid,
	nn.Tanh: QuantTanh,
	nn.LeakyReLU: QuantLeakyReLU,
	nn.SiLU: QuantSiLU,
	nn.SELU: QuantSELU,
	nn.GELU: QuantGELU,
	nn.Identity: QuantIdentity,
}

_CONV_ATTRS = ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups')


def _quantize_layer(layer):
	quant_cls = LAYER_MAP[type(layer)]
	if isinstance(layer, nn.Linear):
		quant_layer = quant_cls(layer.in_features, layer.out_features, bias=layer.bias is not None)
	else:
		kwargs = {name: getattr(layer, name) for name in _CONV_ATTRS}
		quant_layer = quant_cls(bias=layer.bias is not None, **kwargs)
	quant_layer.load_state_dict(layer.state_dict(), strict=False)
	return quant_layer


def _quantize_activation(act):
	return ACT_MAP.get(type(act), QuantIdentity)()


class quantized_model(nn.Module):
	# Rebuilds a plain float nn.Module (compute layer -> activation -> ... ->
	# optional final Softmax, e.g. XOR) as a chain of XBundle-wrapped xlayer quant
	# layers: each compute layer becomes its xlayer Quant* equivalent (trained float
	# weights copied over), paired with the xlayer Quant* equivalent of the
	# activation that follows it as core.act, wrapped in an XBundle.
	def __init__(self, net):
		super().__init__()
		self.bundles = nn.ModuleList()

		children = list(net.children())
		i = 0
		while i < len(children):
			layer = children[i]
			if type(layer) not in LAYER_MAP:
				raise ValueError(
					f"quantized_model does not know how to quantize {type(layer).__name__}; "
					f"expected one of {[t.__name__ for t in LAYER_MAP]} at this point in the net")
			core = _quantize_layer(layer)
			i += 1

			if i < len(children) and type(children[i]) in ACT_MAP:
				core.act = _quantize_activation(children[i])
				i += 1
			else:
				core.act = QuantIdentity()

			softmax = i < len(children) and isinstance(children[i], nn.Softmax)
			if softmax:
				i += 1

			self.bundles.append(XBundle(core=core, softmax=softmax))

	def forward(self, x):
		reset_bundles()
		for bundle in self.bundles:
			x = bundle(x)
		return x
