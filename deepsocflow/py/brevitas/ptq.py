import json
import math

import torch
import torch.nn as nn
from brevitas.quant import Int32Bias
from brevitas.quant.fixed_point import (
	Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Uint8ActPerTensorFixedPoint,
)
from brevitas.graph.calibrate import calibration_mode
from brevitas.export import export_qonnx

from deepsocflow.py.brevitas.utils import reset_bundles
from deepsocflow.py.brevitas.xbundle.xbundle import XBundle
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d
from deepsocflow.py.brevitas.xlayer.quantActivation import (
	QuantIdentity, QuantReLU, QuantSigmoid, QuantTanh,
	QuantLeakyReLU, QuantSiLU, QuantSELU, QuantGELU,
)
from deepsocflow.py.brevitas.xlayer.quantPooling import (
	QuantAvgPool2d, QuantAdaptiveAvgPool2d,
	QuantMaxPool1d, QuantMaxPool2d, QuantMaxPool3d,
	QuantAdaptiveMaxPool1d, QuantAdaptiveMaxPool2d, QuantAdaptiveMaxPool3d,
)

# model json or model.py

# torch -> high precision -> quantization -> export to qonnx
# or brevitas -> low precision -> export to qonnx

# qonnx json

# cgra4ml

_CONV_ATTRS = ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups')


def _frac_bits(scale):
	# scale is power-of-two (2^-frac) by construction - see Int*FixedPoint quantizers.
	return round(-math.log2(scale))


def _act_type_name(act):
	return type(act).__name__.replace("Quant", "").lower()


def _quantize_layer(layer, weight_bits=8, bias_bits=32, own_input_quant=True):
	# Compute layers: plain torch type -> our xlayer quant equivalent.
	LAYER_MAP = {
		nn.Linear: QuantLinear,
		nn.Conv1d: QuantConv1d,
		nn.Conv2d: QuantConv2d,
		nn.Conv3d: QuantConv3d,
	}
	quant_cls = LAYER_MAP.get(type(layer))
	if quant_cls is None:
		return None

	has_bias = layer.bias is not None
	bias_quant = Int32Bias if has_bias else None

	# own_input_quant=False means this layer consumes the previous bundle's
	# activation output directly as an already-quantized QuantTensor (that
	# activation was built with return_quant_tensor=True) instead of
	# re-quantizing it - a single quantization point per bundle, matching
	# qkeras's structure, instead of one after every activation AND one
	# before every layer.
	input_quant = Int8ActPerTensorFixedPoint if own_input_quant else None

	if isinstance(layer, nn.Linear):
		quant_layer = quant_cls(
			layer.in_features, layer.out_features, bias=has_bias,
			weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=weight_bits,
			input_quant=input_quant,
			bias_quant=bias_quant, bias_bit_width=bias_bits)
	else:
		kwargs = {name: getattr(layer, name) for name in _CONV_ATTRS}
		quant_layer = quant_cls(
			bias=has_bias,
			weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=weight_bits,
			input_quant=input_quant,
			bias_quant=bias_quant, bias_bit_width=bias_bits, **kwargs)
	quant_layer.load_state_dict(layer.state_dict(), strict=False)

	quant_layer.configured_bias_bits = bias_bits if has_bias else None
	quant_layer.has_own_input_quant = own_input_quant
	return quant_layer


def _quantize_activation(act):
	# Activations: plain torch type -> (our xlayer quant equivalent, power-of-two scale
	# act_quant matching its sign - Uint8 for ReLU/Sigmoid outputs, which are >= 0,
	# Int8 for everything else).
	ACT_MAP = {
		nn.ReLU: (QuantReLU, Uint8ActPerTensorFixedPoint),
		nn.Sigmoid: (QuantSigmoid, Uint8ActPerTensorFixedPoint),
		nn.Tanh: (QuantTanh, Int8ActPerTensorFixedPoint),
		nn.LeakyReLU: (QuantLeakyReLU, Int8ActPerTensorFixedPoint),
		nn.SiLU: (QuantSiLU, Int8ActPerTensorFixedPoint),
		nn.SELU: (QuantSELU, Int8ActPerTensorFixedPoint),
		nn.GELU: (QuantGELU, Int8ActPerTensorFixedPoint),
		nn.Identity: (QuantIdentity, Int8ActPerTensorFixedPoint),
	}
	entry = ACT_MAP.get(type(act))
	if entry is None:
		return None
	quant_cls, act_quant = entry

	# return_quant_tensor=True: every activation's output must carry its own
	# scale/bit-width so the next bundle's layer (built with
	# own_input_quant=False) can consume it directly instead of re-quantizing.
	kwargs = {"act_quant": act_quant, "return_quant_tensor": True}

	# Unsigned activations (ReLU/Sigmoid) are narrowed to bits-1 so their output
	# still fits the signed datapath every other tensor in this project uses
	# (mirrors xlayers.py:31-33: "QKeras treats relu as unsigned, we have
	# everything signed, so we reduce bitwidth" - an unsigned 8-bit value can
	# reach 255, which silently wraps when packed into a signed 8-bit word).
	if act_quant is Uint8ActPerTensorFixedPoint:
		kwargs["bit_width"] = 7

	return quant_cls(**kwargs)


def _quantize_pool(pool):
	# Pooling: plain torch type -> our xlayer quant equivalent.
	POOL_MAP = {
		nn.MaxPool1d: QuantMaxPool1d,
		nn.MaxPool2d: QuantMaxPool2d,
		nn.MaxPool3d: QuantMaxPool3d,
		nn.AdaptiveMaxPool1d: QuantAdaptiveMaxPool1d,
		nn.AdaptiveMaxPool2d: QuantAdaptiveMaxPool2d,
		nn.AdaptiveMaxPool3d: QuantAdaptiveMaxPool3d,
		nn.AvgPool2d: QuantAvgPool2d,
		nn.AdaptiveAvgPool2d: QuantAdaptiveAvgPool2d,
	}
	quant_cls = POOL_MAP.get(type(pool))
	return quant_cls() if quant_cls is not None else None


class quantized_model(nn.Module):
	# Rebuilds a plain float nn.Module (compute layer -> activation -> ... ->
	# optional final Softmax, e.g. XOR) as a chain of XBundle-wrapped xlayer quant
	# layers: each compute layer becomes its xlayer Quant* equivalent (trained float
	# weights copied over), paired with the xlayer Quant* equivalent of the
	# activation that follows it as core.act, wrapped in an XBundle.
	#
	# weight_bits/bias_bits: default bit width applied to every compute layer's
	# weight/bias. layer_bits: optional {attr_name: {"weight_bits": N, "bias_bits": N}}
	# to override the default for specific layers, keyed by the float net's own
	# attribute name (e.g. {"hidden_1": {"weight_bits": 4}}).
	def __init__(self, net, weight_bits=8, bias_bits=32, layer_bits=None):
		super().__init__()
		self.bundles = nn.ModuleList()
		layer_bits = layer_bits or {}

		named_children = list(net.named_children())
		children = [child for _, child in named_children]
		i = 0
		while i < len(children):
			name, layer = named_children[i]
			overrides = layer_bits.get(name, {})
			core = _quantize_layer(
				layer,
				weight_bits=overrides.get('weight_bits', weight_bits),
				bias_bits=overrides.get('bias_bits', bias_bits),
				own_input_quant=(len(self.bundles) == 0))
			if core is None:
				raise ValueError(
					f"quantized_model does not know how to quantize {type(layer).__name__}; "
					f"expected a compute layer (Linear/ConvNd) at this point in the net")
			i += 1

			quant_act = _quantize_activation(children[i]) if i < len(children) else None
			if quant_act is not None:
				core.act = quant_act
				i += 1
			else:
				core.act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

			softmax = i < len(children) and isinstance(children[i], nn.Softmax)
			if softmax:
				i += 1

			self.bundles.append(XBundle(core=core, softmax=softmax))

		self.print_graph()

	def print_graph(self):
		print("=== quantized_model graph ===")
		for idx, bundle in enumerate(self.bundles):
			core = bundle.core
			shape = f"in={core.in_features} out={core.out_features}" if hasattr(core, "in_features") else ""
			w_bits = int(core.weight_quant.bit_width().item())
			b_bits = core.configured_bias_bits
			print(f"bundle[{idx}]  core={type(core).__name__}({shape})  weight_bits={w_bits}  bias_bits={b_bits}  "
				f"act={type(core.act).__name__}  softmax={bundle.softmax is not None}")

	def forward(self, x):
		reset_bundles()
		for bundle in self.bundles:
			x = bundle(x)
		return x

	# Runs PTQ calibration: forwards calibration_data through the model with
	# calibration_mode enabled so each quant layer's scale is set from real
	# activation statistics instead of brevitas's freshly-initialized default.
	def quantization(self, calibration_data):
		self.eval()
		with torch.no_grad(), calibration_mode(self):
			self(calibration_data)
		return self

	# Exports this (already-quantized, already-calibrated) model to a QONNX file.
	# sample_input: a representative input tensor, only used to trace the graph shape -
	# its values don't matter, but its shape/dtype must match real inference inputs.
	def export(self, sample_input, path):
		self.eval()
		# dynamo=False: brevitas's QONNX export relies on the legacy TorchScript-based
		# torch.onnx.export path (its quant proxies trip up torch.export's dynamo tracer,
		# e.g. DataDependentOutputException on aten.allclose inside scale computation).
		export_qonnx(self, args=sample_input, export_path=path, dynamo=False)
		return path

	# Walks self.bundles and serializes everything a fixed-point simulator needs per
	# layer: hyperparameters, quantized weight/bias INT levels (not float), and
	# bit-width/frac-bits for every tensor role (input, weight, bias, activation
	# output). Call after quantization() - bias_quant needs a real forward's
	# input_scale to resolve, which this reproduces per bundle.
	def export_graph_json(self, sample_input, path):
		self.eval()
		layers = {}
		x = sample_input
		prev_name = None
		with torch.no_grad():
			for idx, bundle in enumerate(self.bundles):
				name = f"bundle{idx}"
				core = bundle.core

				quant_input = core.input_quant(x) if core.has_own_input_quant else x
				quant_weight = core.quant_weight(quant_input)
				quant_bias = core.bias_quant(core.bias, quant_input, quant_weight) if core.bias is not None else None

				input_signed = bool(quant_input.signed)
				weight_signed = bool(quant_weight.signed)
				bias_signed = bool(quant_bias.signed) if quant_bias is not None else None

				w_scale = quant_weight.scale.item()
				w_int = (quant_weight.value / w_scale).round().to(torch.int64).tolist()

				layer = {
					"type": "linear" if hasattr(core, "in_features") else "conv",
					"input": prev_name,
					"input_bits": int(quant_input.bit_width.item()),
					"input_signed": input_signed,
					"input_frac": _frac_bits(quant_input.scale.item()),
					"input_scale": quant_input.scale.item(),
					"input_zero_point": quant_input.zero_point.item(),
					"weight": {
						"bits": int(quant_weight.bit_width.item()),
						"signed": weight_signed,
						"frac": _frac_bits(w_scale),
						"scale": w_scale,
						"zero_point": quant_weight.zero_point.item(),
						"values": w_int,
						"values_float": quant_weight.value.tolist(),
					},
				}
				if hasattr(core, "in_features"):
					layer["in_features"] = core.in_features
					layer["out_features"] = core.out_features

				if quant_bias is not None:
					b_scale = quant_bias.scale.item()
					layer["bias"] = {
						"bits": int(quant_bias.bit_width.item()),
						"signed": bias_signed,
						"frac": _frac_bits(b_scale),
						"scale": b_scale,
						"zero_point": quant_bias.zero_point.item(),
						"values": (quant_bias.value / b_scale).round().to(torch.int64).tolist(),
						"values_float": quant_bias.value.tolist(),
					}

				x = bundle(x)  # advance to this bundle's real (post-act/softmax) output

				layer["activation"] = _act_type_name(core.act)
				if core.act.act_quant.is_quant_enabled:
					act_scale = core.act.act_quant.scale().item()
					layer["act_bits"] = int(core.act.act_quant.bit_width().item())
					layer["act_frac"] = _frac_bits(act_scale)
					layer["act_scale"] = act_scale
					layer["act_zero_point"] = core.act.act_quant.zero_point().item()
					layer["act_signed"] = bool(core.act.act_quant.is_signed)
				layer["softmax"] = bundle.softmax is not None

				layers[name] = layer
				prev_name = name

		with open(path, 'w') as f:
			json.dump({"layers": layers}, f, indent=2)
		return path
