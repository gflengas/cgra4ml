import json
import math
import types

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
	QuantAvgPool2d, QuantAdaptiveAvgPool2d, QuantAvgPool2dDivRound,
	QuantMaxPool2d, QuantAdaptiveMaxPool2d,
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


def _as_pair(v):
	return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def tf_same_padding(n, k, s):
    """(pad_lo, pad_hi) reproducing TF's 'same' padding on one axis.

    Public because model authors need it: a strided conv can only match the
    engine if the float model pads exactly this way, and hand-computing it is
    how the off-by-one gets in. Use it as
    `nn.ZeroPad2d((left, right, top, bottom))` before a `padding=0` conv.

    The smaller half goes to the top/left, which is what makes it asymmetric for
    an even input - and precisely what torch's `Conv2d(padding=)` cannot express.
    """
    out = -(-n // s)  # ceil
    total = max((out - 1) * s + k - n, 0)
    lo = total // 2
    return lo, total - lo


def _conv_geometry(core, explicit_pad=None):
	# Normalizes a QuantConv2d's hyperparameters into the flat form the graph
	# JSON carries, and enforces the two shapes the hardware can actually run.
	#
	# The engine always computes a stride-1, SAME-padded convolution over the
	# whole input; conv striding is the CPU dropping output pixels afterwards
	# (runtime.h's CONV STRIDING block, driven by dataflow.py's CSH/CSW). So the
	# padding that has to be SAME is the padding of that underlying stride-1
	# convolution - which for an odd kernel is kernel_size//2. torch spells the
	# same thing two ways depending on stride (it rejects padding='same' on a
	# strided conv), so both spellings are accepted and normalized here.
	kh, kw = _as_pair(core.kernel_size)
	sh, sw = _as_pair(core.stride)
	dh, dw = _as_pair(core.dilation)

	assert (dh, dw) == (1, 1), (
		f"conv dilation {(dh, dw)} is not supported - the engine's weight layout "
		f"(dataflow.py's reorder_w_q2e_conv) has no notion of dilation")
	assert core.groups == 1, (
		f"conv groups={core.groups} is not supported - the engine convolves every "
		f"input channel against every output channel")
	assert kh % 2 == 1 and kw % 2 == 1, (
		f"conv kernel_size {(kh, kw)} must be odd on both axes - 'same' padding is "
		f"only well defined for odd kernels, and dataflow.py's CSH_SHIFT/CSW_SHIFT "
		f"assume it")

	if explicit_pad is not None:
		# The padding lives in a ZeroPad2d ahead of the conv, so the conv itself
		# must not pad again. The geometry still reports the effective 'same'
		# padding, because that is what the engine does internally - the explicit
		# pad exists only to make the float model agree with it.
		assert _as_pair(core.padding) == (0, 0), (
			f"conv has both an explicit pad in front of it and padding={core.padding} "
			f"of its own; the explicit pad is meant to replace it, so the conv must "
			f"use padding=0")
		ph, pw = kh // 2, kw // 2
		return {
			"in_channels": int(core.in_channels),
			"out_channels": int(core.out_channels),
			"kernel_size": [int(kh), int(kw)],
			"stride": [int(sh), int(sw)],
			"padding": [int(ph), int(pw)],
			"explicit_pad": [int(v) for v in explicit_pad],
		}

	padding = core.padding
	if isinstance(padding, str):
		assert padding == 'same', (
			f"conv padding='{padding}' is not supported; the engine only implements "
			f"'same' padding (rtl_export.py::_conv2d_same)")
		ph, pw = kh // 2, kw // 2
	else:
		ph, pw = _as_pair(padding)
		assert (ph, pw) == (kh // 2, kw // 2), (
			f"conv padding {(ph, pw)} is not 'same' for kernel {(kh, kw)} - expected "
			f"{(kh // 2, kw // 2)}. The engine only implements 'same' padding "
			f"(rtl_export.py::_conv2d_same); any other padding would run and produce "
			f"silently wrong output rather than fail.")

	return {
		"in_channels": int(core.in_channels),
		"out_channels": int(core.out_channels),
		"kernel_size": [int(kh), int(kw)],
		"stride": [int(sh), int(sw)],
		"padding": [int(ph), int(pw)],
	}


def _assert_explicit_pad_matches_engine(geom, in_h, in_w):
    # An explicit pad is only correct if it is exactly TF's split; anything else
    # shifts every feature map without changing its shape.
    left, right, top, bottom = geom['explicit_pad']
    kh, kw = geom['kernel_size']
    sh, sw = geom['stride']
    for axis, n, k, s, lo, hi in (('height', in_h, kh, sh, top, bottom),
                                  ('width', in_w, kw, sw, left, right)):
        want_lo, want_hi = tf_same_padding(n, k, s)
        assert (lo, hi) == (want_lo, want_hi), (
            f"explicit pad on the {axis} is ({lo},{hi}) but the engine's 'same' "
            f"padding for size {n}, kernel {k}, stride {s} is ({want_lo},{want_hi}). "
            f"Use ptq.tf_same_padding({n}, {k}, {s}) instead of hand-computing it.")


def _assert_stride_matches_engine(geom, in_h, in_w):
	# torch and the engine disagree about where a STRIDED window starts, and the
	# disagreement is silent: same output shape, plausible values, every feature
	# map shifted by one pixel.
	#
	# The engine follows TF's 'same', which pads by
	# max((ceil(n/s)-1)*s + k - n, 0) and puts the smaller half on the top/left -
	# so for k=3, s=2, n=8 it pads (0, 1) and the first window is centred on row
	# 1. torch's Conv2d cannot express asymmetric padding through its `padding`
	# argument, so a stride-2 layer written the natural way pads (1, 1) and
	# centres its first window on row 0 instead.
	#
	# The two coincide exactly when TF's total padding equals torch's k-1, which
	# for k=3, s=2 means an odd input. Rather than let a mismatch through to be
	# discovered as a wrong accuracy number, refuse it here and name the sizes.
	for axis, n, k, s in (('height', in_h, geom['kernel_size'][0], geom['stride'][0]),
	                      ('width', in_w, geom['kernel_size'][1], geom['stride'][1])):
		if s == 1:
			continue
		out = -(-n // s)  # ceil
		tf_pad_total = max((out - 1) * s + k - n, 0)
		assert tf_pad_total == k - 1, (
			f"strided conv on a {axis} of {n} with kernel {k}, stride {s} cannot be "
			f"expressed with torch's symmetric padding: the engine ('same', TF "
			f"semantics) pads {tf_pad_total} total and starts at "
			f"{(k - 1) // 2 - max((s * (out - 1) + k - n) // 2, 0)}, torch pads {k - 1} "
			f"and starts at 0. Use an input size where they agree (total padding "
			f"== {k - 1}, e.g. an odd size for k=3/s=2), or pad explicitly before "
			f"the conv.")


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _fold_batchnorm(layer, bn):
	"""conv/linear + BatchNorm -> a single equivalent layer, folded.

	The engine has no notion of batch norm at all - `grep -i batchnorm` over
	dataflow.py, rtl_export.py and runtime.h finds nothing, and the datapath is
	conv -> +bias -> shift -> activation. So BN has to be folded host-side; this
	is not an optimization, it is the only way to express it.

	Folded BEFORE quantization, matching what the qkeras path does
	(XConvBN.call_int quantizes get_folded_weights()'s output, xlayers.py:96-97).
	Folding after quantizing would quantize a weight the model never uses.

	torch's fusion helpers compute exactly W*gamma/sqrt(var+eps) and
	(b-mean)*gamma/sqrt(var+eps)+beta, and synthesize a bias when the layer had
	none (fusion.py:85-86) - which is the normal case, since a conv followed by
	BN is almost always built with bias=False. That new bias then flows through
	_quantize_layer's ordinary has_bias path.

	Note this folds ONCE, whereas qkeras's QConv2DBatchnorm re-folds on every
	call. The two are identical whenever the BN is in eval mode with frozen
	running statistics, which PTQ always is; they diverge only under QAT, which
	this backend does not do.
	"""
	from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

	if isinstance(layer, nn.Linear):
		return fuse_linear_bn_eval(layer, bn)
	return fuse_conv_bn_eval(layer, bn)


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


# Activations with no shift-and-clip closed form. On hardware these are executed
# as a value LUT (deepsocflow/py/brevitas/lut.py), and only these are eligible for
# the input_quant that variant 1b needs - relu/leaky_relu/identity keep running on
# quant_lrelu, so giving them an input quantizer would add a quantization point
# that buys nothing.
CURVED_ACT_TYPES = (nn.Sigmoid, nn.Tanh, nn.SiLU, nn.SELU, nn.GELU)


def _quantize_activation(act, act_input_bits=None, share_act_quant=None):
	# Activations: plain torch type -> (our xlayer quant equivalent, power-of-two scale
	# act_quant matching its sign - Uint8 for ReLU/Sigmoid outputs, which are >= 0,
	# Int8 for everything else).
	if isinstance(act, nn.LeakyReLU):
		# quant_lrelu (deepsocflow/c/runtime.h:153) implements the negative-side
		# scaling as a pure left-shift, so it can only realize slopes that are
		# exact negative powers of two - mirrors xlayers.py:23's assert. Without
		# this check a non-power-of-two slope trains and calibrates fine but
		# produces a model with no valid quant_lrelu parameterization, and the
		# failure would only surface much later at RTL-adapter time.
		slope = act.negative_slope
		log_slope = math.log2(slope) if slope > 0 else float("-inf")
		assert slope > 0 and int(log_slope) == log_slope and log_slope <= 0, (
			f"negative_slope={slope} of LeakyReLU must be a negative power of two "
			f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")

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

	# share_act_quant: an already-built activation's quant PROXY, reused so this
	# activation resolves to the SAME scale (and therefore the same frac). The
	# residual path needs that because the hardware adds its two operands raw -
	# Bundle_t carries no add_val_shift/add_a_shift - so they must already share a
	# fractional grid. brevitas supports it via QuantProxyMixin.__init__, which
	# accepts a proxy instance in place of a quantizer class and registers this
	# module on it (add_tracked_module).
	#
	# Note the proxy owns the nonlinearity too (QuantNonLinearActMixin injects
	# act_impl as a quantizer kwarg), so sharing one means sharing the activation
	# function - hence the caller asserts both ends are the same type. Every other
	# kwarg is dropped here on purpose: brevitas warns that keyword arguments are
	# ignored when a built proxy is passed, and bit_width in particular comes from
	# the proxy.
	if share_act_quant is not None:
		return quant_cls(act_quant=share_act_quant, return_quant_tensor=True)

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

	# Variant 1b: quantize the activation's INPUT as well as its output.
	#
	# Without this, brevitas evaluates the activation on the full-precision
	# accumulator while the hardware evaluates it on a table indexed by a
	# shifted-down accumulator - so the two can only agree if the table indexes at
	# the accumulator's own frac, which measures out at 64-128 KB per activation
	# (see lut_poc.py's sweep). Adding input_quant makes brevitas round the
	# accumulator first, exactly as the hardware does, and a 2**act_input_bits
	# entry table then reproduces it bit-exactly.
	#
	# The approximation does not disappear - it moves from "hardware silently
	# disagrees with the model" to "the model itself is coarser", where
	# calibration measures it and QAT can train against it. This does add a second
	# quantization point to the bundle, which the 2026-08-10 single-quantization-
	# point design deliberately avoided; it is opt-in for that reason.
	if act_input_bits is not None and isinstance(act, CURVED_ACT_TYPES):
		kwargs["input_quant"] = Int8ActPerTensorFixedPoint
		kwargs["input_bit_width"] = act_input_bits

	return quant_cls(**kwargs)


def _quantize_pool(pool):
	# Pooling: plain torch type -> our xlayer quant equivalent.
	#
	# Only 2-D max pooling is wired up. Average pooling is deliberately left out:
	# it accumulates, so it would have to agree bit-for-bit with runtime.h's
	# div_round and its widened accumulator (bits + ceil(log2(PKH*PKW)), see
	# xlayers.py:342), while brevitas's TruncAvgPool2d truncates on a different
	# rule. Max just selects an existing value, so there is no rounding behaviour
	# to reconcile. The 1-D/3-D and adaptive variants have no engine equivalent
	# at all - dataflow.py only ever reads a 2-D pool_size/strides/padding.
	POOL_MAP = {
		nn.MaxPool2d: (QuantMaxPool2d, 'max'),
		nn.AvgPool2d: (QuantAvgPool2dDivRound, 'avg'),
	}
	entry = POOL_MAP.get(type(pool))
	if entry is None:
		return None
	quant_cls, pool_type = entry

	kh, kw = _as_pair(pool.kernel_size)
	sh, sw = _as_pair(pool.stride if pool.stride is not None else pool.kernel_size)
	ph, pw = _as_pair(pool.padding)

	assert (ph, pw) == (0, 0), (
		f"pool padding {(ph, pw)} is not supported - the engine implements "
		f"'valid' pooling (dataflow.py:70-72) and 'same', but torch's symmetric "
		f"padding maps cleanly onto neither; use padding=0")
	assert not getattr(pool, 'ceil_mode', False), (
		"pool ceil_mode=True is not supported - dataflow.py's 'valid' output size "
		"((YH-PKH+PSH)//PSH) floors, matching ceil_mode=False")

	if pool_type == 'avg':
		# NOT brevitas's QuantAvgPool2d (TruncAvgPool2d): that sums and truncates to
		# a bit width - a shift - while the engine sums and applies runtime.h's
		# div_round. See QuantAvgPool2dDivRound in quantPooling.py.
		quant_pool = quant_cls(kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))
	else:
		quant_pool = quant_cls(kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))

	# The attribute surface dataflow.py and rtl_export.py read off a pool. Legacy
	# XPool carries a real keras layer here (xlayers.py:259-262); the engine only
	# ever reads these three fields off it, so a namespace is enough.
	quant_pool.type = pool_type
	quant_pool.pool_layer = types.SimpleNamespace(
		pool_size=(kh, kw), strides=(sh, sw), padding='valid')
	# XBundle.call applies pool.act after the pool (xbundle.py:45), mirroring the
	# engine's separate pa_* activation slot. Max pooling returns a value that is
	# already on the incoming activation's grid, so the identity is not a
	# placeholder here - there is genuinely nothing to rescale. nn.Identity also
	# passes the QuantTensor through untouched, keeping its scale/bit-width for
	# the next bundle to consume.
	quant_pool.act = nn.Identity()
	return quant_pool


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
	#
	# act_input_bits: None (default) leaves curved activations evaluated on the
	# full-precision accumulator - the hardware LUT then only approximates them.
	# Setting it (e.g. 8) quantizes each curved activation's input too, which is
	# what makes a 2**act_input_bits entry LUT bit-exact. See _quantize_activation.
	#
	# residuals: optional {consumer_attr: source_attr} declaring skip connections,
	# keyed by the float net's own attribute names (e.g. {"conv_2": "conv_1"} adds
	# conv_1's bundle output into conv_2's bundle). The builder walks
	# net.named_children() linearly and cannot see branches, and torch.fx tracing
	# is not an option here (brevitas's quant proxies break dynamo - see export()),
	# so the topology is declared rather than inferred. Everything downstream
	# consumes the exported JSON, so this can be replaced by real tracing later
	# without touching anything else.
	def __init__(self, net, weight_bits=8, bias_bits=32, layer_bits=None,
	             act_input_bits=None, residuals=None):
		super().__init__()
		self.bundles = nn.ModuleList()
		layer_bits = layer_bits or {}
		self.act_input_bits = act_input_bits
		residuals = residuals or {}
		self.skip_source = {}   # consumer bundle index -> source bundle index
		_bundle_of_attr = {}    # float net attribute name -> bundle index

		named_children = list(net.named_children())
		children = [child for _, child in named_children]
		i = 0
		while i < len(children):
			# An nn.ZeroPad2d ahead of the compute layer carries TF's asymmetric
			# 'same' padding for a strided conv (see tf_same_padding). It is kept on
			# the bundle rather than consumed, because the float model has to apply
			# it - the engine still gets the unpadded tensor and pads internally.
			pre_pad = None
			if isinstance(children[i], nn.ZeroPad2d):
				pre_pad = children[i]
				i += 1

			name, layer = named_children[i]
			overrides = layer_bits.get(name, {})

			# A BatchNorm directly after the compute layer is folded into it and
			# consumed here; one anywhere else still falls through to the "does not
			# know how to quantize" error below, which is correct - there is nothing
			# to fold it into.
			if i + 1 < len(children) and isinstance(children[i + 1], _BN_TYPES):
				# Only demanded where it actually matters: in train mode a BatchNorm
				# normalizes by the batch's own statistics rather than its running
				# ones, so folding it - and calibrating against it - would describe a
				# model that is not the one being deployed. Nets without BN are left
				# alone rather than being made to care.
				assert not net.training and not children[i + 1].training, (
					f"'{name}' is followed by {type(children[i + 1]).__name__} but the "
					f"net is in train mode - call net.eval() first. A BatchNorm in "
					f"train mode uses batch statistics instead of its running ones, so "
					f"folding it would quantize a different model than the one that "
					f"gets deployed.")
				layer = _fold_batchnorm(layer, children[i + 1])
				i += 1

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

			ib = len(self.bundles)
			_bundle_of_attr[name] = ib

			# A residual consumer must land its core activation on the SAME
			# fractional grid as the bundle it adds, because the hardware adds the
			# two raw (Bundle_t has no add_val_shift/add_a_shift). Sharing the
			# source's act quant proxy is what guarantees it; the assert below is
			# what stops a silently-misaligned pair, which would be wrong by a
			# power of two while still looking plausible.
			share_act_quant = None
			source_attr = residuals.get(name)
			if source_attr is not None:
				assert source_attr in _bundle_of_attr, (
					f"residual source '{source_attr}' for '{name}' is not a compute layer "
					f"seen earlier in the net; a skip can only come from a bundle that "
					f"has already been built")
				src_ib = _bundle_of_attr[source_attr]
				self.skip_source[ib] = src_ib
				src_act = self.bundles[src_ib].core.act
				consumer_act = children[i] if i < len(children) else None
				assert consumer_act is not None and type(src_act).__name__ == \
					'Quant' + type(consumer_act).__name__, (
					f"residual pair '{source_attr}' -> '{name}' must use the same "
					f"activation: the shared quant proxy owns the nonlinearity, so the "
					f"consumer's own would be ignored. Source has "
					f"{type(src_act).__name__}, consumer has "
					f"{type(consumer_act).__name__ if consumer_act else None}.")
				share_act_quant = src_act.act_quant

			quant_act = _quantize_activation(
				children[i], act_input_bits, share_act_quant=share_act_quant
			) if i < len(children) else None
			if quant_act is not None:
				core.act = quant_act
				i += 1
			else:
				assert source_attr is None, (
					f"residual consumer '{name}' has no activation after it; the shared "
					f"grid is established through the core activation")
				core.act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)

			# Pool and flatten sit between the activation and the softmax, in the
			# order runtime.h applies them (CORE ACT -> residual -> POOLING, then
			# tile_write's flatten). XBundle.call already threads them; the build
			# loop simply never looked for them before.
			pool = _quantize_pool(children[i]) if i < len(children) else None
			if pool is not None:
				i += 1
				# An activation after the pool would be silently dropped, not
				# applied: runtime.h never uses pa_* for max pooling at all, and for
				# average pooling it applies quant_lrelu to `out_val` (the pre-pool
				# pixel) and then writes `result` - so the activation's own output is
				# discarded. Refuse it here rather than let the model claim a
				# nonlinearity the hardware will not run.
				assert i >= len(children) or _quantize_activation(children[i]) is None, (
					f"'{name}' has an activation after its pooling layer "
					f"({type(children[i]).__name__}); the engine has no working "
					f"post-pool activation slot, so it would be ignored. Put the "
					f"activation before the pool.")

			flatten = i < len(children) and isinstance(children[i], nn.Flatten)
			if flatten:
				i += 1

			softmax = i < len(children) and isinstance(children[i], nn.Softmax)
			if softmax:
				i += 1

			# The add activation requantizes the sum back onto the activation grid
			# (the sum needs one more bit than either operand). It is a plain
			# identity: runtime.h:452 hardwires quant_lrelu for this slot and
			# Bundle_t has no aa_lut_idx, so a curved activation here is not
			# expressible at all.
			add_act = QuantIdentity(
				act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True
			) if source_attr is not None else None

			self.bundles.append(XBundle(core=core, pool=pool, flatten=flatten,
			                            softmax=softmax, add_act=add_act,
			                            pre_pad=pre_pad))

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
		# Outputs are kept per bundle rather than threaded through one variable,
		# because a skip consumer needs an earlier bundle's output as well as its
		# immediate predecessor's. XBundle.call already accepts x_add.
		outs = []
		for idx, bundle in enumerate(self.bundles):
			src = self.skip_source.get(idx)
			x = bundle(x, outs[src]) if src is not None else bundle(x)
			outs.append(x)
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
		# Same reason forward() keeps them: a skip consumer needs an earlier
		# bundle's output, not just its predecessor's.
		outs = []
		with torch.no_grad():
			for idx, bundle in enumerate(self.bundles):
				name = f"bundle{idx}"
				core = bundle.core
				src_ib = self.skip_source.get(idx)

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
				else:
					layer.update(_conv_geometry(core, explicit_pad=(
						tuple(bundle.pre_pad.padding) if bundle.pre_pad is not None else None)))
					# The spatial shape this bundle receives. sim.py cannot derive
					# it from the weights the way it can for a dense layer, and the
					# engine layout (blocks, passes, X_PAD) is computed from it -
					# so it has to be recorded rather than inferred. NCHW, matching
					# torch; sim.py transposes to the NHWC the exporter wants.
					layer["input_shape"] = [int(d) for d in quant_input.value.shape]
					if 'explicit_pad' in layer:
						_assert_explicit_pad_matches_engine(
							layer, layer["input_shape"][2], layer["input_shape"][3])
					else:
						_assert_stride_matches_engine(
							layer, layer["input_shape"][2], layer["input_shape"][3])

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

				x = bundle(x, outs[src_ib]) if src_ib is not None else bundle(x)
				outs.append(x)  # advance to this bundle's real (post-act/softmax) output

				layer["activation"] = _act_type_name(core.act)
				if core.act.act_quant.is_quant_enabled:
					act_scale = core.act.act_quant.scale().item()
					layer["act_bits"] = int(core.act.act_quant.bit_width().item())
					layer["act_frac"] = _frac_bits(act_scale)
					layer["act_scale"] = act_scale
					layer["act_zero_point"] = core.act.act_quant.zero_point().item()
					layer["act_signed"] = bool(core.act.act_quant.is_signed)

				# Variant 1b only: the grid the activation's input was quantized
				# onto. A LUT indexed on exactly this grid reproduces brevitas's
				# own output bit-exactly, so sim.py sizes the table from these
				# rather than guessing. Absent for 1a models, where brevitas
				# evaluated the activation on the raw accumulator.
				act_in_quant = getattr(core.act, 'input_quant', None)
				if act_in_quant is not None and act_in_quant.is_quant_enabled:
					act_in_scale = act_in_quant.scale().item()
					layer["act_in_bits"] = int(act_in_quant.bit_width().item())
					layer["act_in_frac"] = _frac_bits(act_in_scale)
					layer["act_in_scale"] = act_in_scale
					layer["act_in_signed"] = bool(act_in_quant.is_signed)

				# Pool runs after the core activation and before flatten, matching
				# runtime.h's order. Only the three fields dataflow.py reads are
				# emitted; sim.py reconstructs the rest from them.
				if bundle.pool is not None:
					layer["pool"] = {
						"type": bundle.pool.type,
						"size": list(bundle.pool.pool_layer.pool_size),
						"strides": list(bundle.pool.pool_layer.strides),
						"padding": bundle.pool.pool_layer.padding,
					}
				# Residual: the bundle whose output is added in, plus the activation
				# applied to the sum. Named "skip_from" to match the field CLAUDE.md's
				# Known Issues already reserved for it.
				if src_ib is not None:
					layer["skip_from"] = f"bundle{src_ib}"
					add_act = bundle.add.act
					layer["add_activation"] = _act_type_name(add_act)
					if add_act.act_quant.is_quant_enabled:
						add_scale = add_act.act_quant.scale().item()
						layer["add_act_bits"] = int(add_act.act_quant.bit_width().item())
						layer["add_act_frac"] = _frac_bits(add_scale)
						layer["add_act_scale"] = add_scale
						layer["add_act_signed"] = bool(add_act.act_quant.is_signed)

				layer["flatten"] = bundle.flatten is not None
				layer["softmax"] = bundle.softmax is not None

				layers[name] = layer
				prev_name = name

		with open(path, 'w') as f:
			json.dump({"layers": layers}, f, indent=2)
		return path
