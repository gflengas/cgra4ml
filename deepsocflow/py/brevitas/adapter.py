"""Adapts brevitas FixedPointModel bundles onto the legacy XBundle attribute
surface, so the legacy engine-layout export (deepsocflow/py/xmodel.py) and RTL
verification path run over brevitas-produced numbers unchanged.

brevitas owns the numbers; the legacy backend owns the file format. This module
is the only seam between them."""
import math
import numpy as np


def act_params(activation, negative_slope=0.0):
    """(non_zero, plog_slope) as legacy XActivation computes them
    (deepsocflow/py/xlayers.py:20-24).

    non_zero is 0 only for plain relu (slope 0); identity is modelled by legacy
    as slope=1, which makes non_zero 1 and plog_slope 0. plog_slope is the
    right-shift amount applied to negative inputs, so it is only non-zero for
    leaky_relu."""
    if activation == 'relu':
        return 0, 0
    if activation == 'identity':
        return 1, 0
    if activation == 'leaky_relu':
        log_slope = math.log2(negative_slope)
        assert log_slope == int(log_slope) and log_slope <= 0, (
            f"negative_slope={negative_slope} must be a negative power of two "
            f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")
        return 1, -int(log_slope)
    raise NotImplementedError(
        f"activation '{activation}' has no integer-exact hardware implementation "
        f"(see CLAUDE.md Known Issues); only relu/identity/leaky_relu are deployable")


def to_engine_weight(weight_int):
    """torch Linear weight (out_features, in_features) -> legacy conv weight
    (KH, KW, CI, CO) = (1, 1, in_features, out_features).

    The transpose is real: torch stores (out, in), keras stores (in, out)."""
    return np.asarray(weight_int).T[None, None, :, :]


def to_engine_activation(x_int):
    """(batch, features) -> (XN, XH, XW, CI) = (1, batch, 1, features).

    Batch goes in the H slot, not the N slot - this mirrors the legacy dense
    reshape at xbundle.py:126. Getting it backwards produces wrong runtime
    params (XL, X_PAD) without any error."""
    return np.asarray(x_int)[None, :, None, :]


from deepsocflow.py.utils import BUNDLES, XTensor


class _Act:
    """Stands in for legacy XActivation. Only the four attributes the export
    path reads are provided - there is no call_int, because the adapter never
    recomputes anything."""

    def __init__(self, non_zero, plog_slope, shift_bits, out):
        self.non_zero = non_zero
        self.plog_slope = plog_slope
        self.shift_bits = shift_bits
        self.out = out


class _Core:
    """Stands in for legacy XDense. type/strides/padding are what
    get_runtime_params reads off a dense core (dataflow.py:34-47)."""

    type = 'dense'
    strides = (1, 1)
    padding = 'same'

    def __init__(self, w, x, y, b, act):
        self.w, self.x, self.y, self.b, self.act = w, x, y, b, act


class BrevitasBundle:
    """One brevitas bundle wearing legacy XBundle's attribute surface."""

    def __init__(self, ib, core, softmax, out, pre_softmax, prev_ib):
        self.ib = ib
        self.core = core
        self.pool = None
        self.add = None
        self.flatten = False
        self.softmax = softmax
        self.out = out
        self.pre_softmax = pre_softmax
        self.prev_ib = prev_ib
        self.next_ibs = set()
        self.next_add_ibs = set()

    def call_int(self, x, hw):
        """No-op: brevitas already computed every integer tensor and the adapter
        pre-populated them. Legacy XBundle.call_int recomputes the bundle in
        integer arithmetic; doing that here would either duplicate sim.py or
        silently disagree with it."""
        return self.out

    def export(self, hw, is_last):
        from deepsocflow.py.xbundle import XBundle
        return XBundle.export(self, hw, is_last)


def build_bundles(model, hw, has_bias=None):
    """Builds one BrevitasBundle per FixedPointModel bundle, wires the chain
    topology, and registers them into the legacy BUNDLES global (replacing
    whatever was there).

    model must have had forward() called already - the adapter reads .trace.
    has_bias maps bundle name -> bool; defaults to True for every bundle, since
    the JSON exporter only omits "bias" when the layer genuinely has none."""
    has_bias = {} if has_bias is None else has_bias

    for b in BUNDLES:
        b.next_ibs.clear()
        b.next_add_ibs.clear()
    BUNDLES.clear()

    index_of = {name: i for i, name in enumerate(model.bundle_order)}
    adapters = []

    for ib, name in enumerate(model.bundle_order):
        cfg = model.bundles[name]
        trace = model.trace[name]

        acc_frac = cfg['input_frac'] + cfg['weight_frac']
        non_zero, plog_slope = act_params(cfg['activation'])

        act_out = XTensor(
            tensor=np.asarray(trace['out'], dtype=np.float32),
            bits=cfg['act_bits'], frac=cfg['act_frac'], from_int=True)
        act = _Act(
            non_zero=non_zero,
            plog_slope=plog_slope,
            shift_bits=plog_slope + acc_frac - cfg['act_frac'],
            out=act_out)

        w = XTensor(
            tensor=to_engine_weight(cfg['weight']).astype(np.float32),
            bits=hw.K_BITS, frac=cfg['weight_frac'], from_int=True)
        x = XTensor(
            tensor=to_engine_activation(trace['x']).astype(np.float32),
            bits=cfg['input_bits'], frac=cfg['input_frac'], from_int=True)
        y = XTensor(
            tensor=to_engine_activation(trace['y']).astype(np.float32),
            bits=hw.Y_BITS, frac=acc_frac, from_int=True)

        if has_bias.get(name, True):
            b = XTensor(tensor=np.asarray(cfg['bias'], dtype=np.float32),
                        bits=hw.B_BITS, frac=cfg['bias_frac'], from_int=True)
        else:
            b = None

        is_last = ib == len(model.bundle_order) - 1
        if is_last and cfg['softmax']:
            pre_softmax = XTensor(
                tensor=to_engine_activation(model.pre_softmax).astype(np.float32),
                bits=cfg['act_bits'], frac=model.softmax_frac, from_int=True)
            out = XTensor(
                tensor=to_engine_activation(model.softmax_out).astype(np.float32),
                bits=None, float_only=True)
        else:
            pre_softmax = None
            out = act_out

        prev_ib = index_of[cfg['input']] if cfg['input'] is not None else None
        adapter = BrevitasBundle(
            ib=ib,
            core=_Core(w=w, x=x, y=y, b=b, act=act),
            softmax=bool(cfg['softmax']),
            out=out,
            pre_softmax=pre_softmax,
            prev_ib=prev_ib)

        adapters.append(adapter)
        BUNDLES.append(adapter)

    for adapter in adapters:
        if adapter.prev_ib is not None:
            adapters[adapter.prev_ib].next_ibs.add(adapter.ib)

    return adapters
