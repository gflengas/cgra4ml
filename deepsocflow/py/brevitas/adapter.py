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
        assert negative_slope > 0, (
            f"negative_slope={negative_slope} must be a negative power of two "
            f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")
        log_slope = math.log2(negative_slope)
        assert log_slope == int(log_slope) and log_slope <= 0, (
            f"negative_slope={negative_slope} must be a negative power of two "
            f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")
        return 1, -int(log_slope)
    raise NotImplementedError(
        f"activation '{activation}' has no integer-exact hardware implementation "
        f"(see CLAUDE.md Known Issues); only relu/identity/leaky_relu are deployable")


def to_legacy_dense_weight(weight_int):
    """torch Linear weight (out_features, in_features) -> legacy dense weight
    (CI, CO) = (in_features, out_features).

    Only the transpose happens here. The reshape to the 4-D engine layout
    (1, 1, CI, CO) is done by XBundle.export's own dense branch (xbundle.py:139),
    which also reshapes x, y and the output - so everything this adapter hands to
    the legacy path must stay 2-D."""
    return np.asarray(weight_int).T


from deepsocflow.py.numeric import BUNDLES
from deepsocflow.py.brevitas.xtensor import XTensor


class _Act:
    """Stands in for legacy XActivation. Only the four attributes the export
    path reads are provided - there is no call_int, because the adapter never
    recomputes anything."""

    def __init__(self, non_zero, plog_slope, shift_bits, out, lut=None):
        self.non_zero = non_zero
        self.plog_slope = plog_slope
        self.shift_bits = shift_bits
        self.out = out
        # An ActLut (deepsocflow/py/brevitas/lut.py) for curved activations,
        # None for the quant_lrelu ones. rtl_export.py reads it to emit the
        # table and .ca_lut_idx; nothing in the legacy path touches it.
        self.lut = lut


class _Core:
    """Stands in for legacy XDense. type/strides/padding are what
    get_runtime_params reads off a dense core (dataflow.py:34-47)."""

    type = 'dense'
    strides = (1, 1)
    padding = 'same'

    # Read by config_fw.h's writer (xmodel.py:234). Legacy derives them in
    # XDense.call_int via out.add_val_shift(self.b), which the adapter's inert
    # call_int never runs. add_val_shift (utils.py:66-81) returns
    # (max(y.frac,b.frac)-y.frac, max(y.frac,b.frac)-b.frac); brevitas's single
    # quantization point per bundle makes bias_frac == acc_frac (sim.py:166
    # asserts it), so both shifts are identically zero.
    bias_val_shift = 0
    bias_b_shift = 0

    def __init__(self, w, x, y, b, act):
        self.w, self.x, self.y, self.b, self.act = w, x, y, b, act


class BrevitasBundle:
    """One brevitas bundle wearing legacy XBundle's attribute surface."""

    def __init__(self, ib, core, softmax, out, pre_softmax, prev_ib):
        self.ib = ib
        self.core = core
        self.pool = None
        self.add = None
        # None-vs-truthy matters: xmodel.py:233 emits is_flatten/is_softmax with
        # `is not None`, while xbundle.py:144 and xmodel.py:221 test truthiness.
        # Legacy stores None when absent (xbundle.py:41,44), so `False` here would
        # make every bundle claim to be flattened and softmaxed.
        self.flatten = None
        self.softmax = True if softmax else None
        self.out = out
        self.pre_softmax = pre_softmax
        self.prev_ib = prev_ib
        self.next_ibs = set()
        self.next_add_ibs = set()

        # Read by config_fw.h's writer (xmodel.py:234). Legacy defaults both to 0
        # (xbundle.py:47-48) and fills them in inside call_int, which this
        # adapter deliberately no-ops - so build_bundles sets them instead.
        self.softmax_frac = 0
        self.softmax_max_i = 0

    def call_int(self, x, hw):
        """No-op: brevitas already computed every integer tensor and the adapter
        pre-populated them. Legacy XBundle.call_int recomputes the bundle in
        integer arithmetic; doing that here would either duplicate sim.py or
        silently disagree with it."""
        return self.out

    def export(self, hw, is_last):
        from deepsocflow.py.brevitas.rtl_export import export_bundle
        return export_bundle(self, hw, is_last)


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
        lut = cfg.get('lut')

        act_out = XTensor(
            tensor=np.asarray(trace['out'], dtype=np.float32),
            bits=cfg['act_bits'], frac=cfg['act_frac'], from_int=True)

        if lut is not None:
            # shift_bits changes MEANING on a LUT bundle. On the quant_lrelu path
            # it is the shift onto the activation's OUTPUT grid, because that is
            # where the shift lands the value. On the LUT path the shift lands on
            # the table's INDEX grid instead, and the output grid is reached by the
            # table itself - so the target is lut.in_frac, not cfg['act_frac'].
            #
            # These coincide only under variant 1a (index grid == output grid), so
            # using act_frac here would still produce a plausible-looking, running,
            # wrong result on every 1b model. non_zero/plog_slope are unused on
            # this path; they are set to the identity values so that a config_fw.h
            # dump reads as "no lrelu behaviour" rather than as leftovers.
            non_zero, plog_slope = 1, 0
            shift_bits = lut.index_shift(acc_frac)
        else:
            non_zero, plog_slope = act_params(cfg['activation'])
            shift_bits = plog_slope + acc_frac - cfg['act_frac']

        act = _Act(
            non_zero=non_zero,
            plog_slope=plog_slope,
            shift_bits=shift_bits,
            out=act_out,
            lut=lut)

        w = XTensor(
            tensor=to_legacy_dense_weight(cfg['weight']).astype(np.float32),
            bits=hw.K_BITS, frac=cfg['weight_frac'], from_int=True)
        x = XTensor(
            tensor=np.asarray(trace['x'], dtype=np.float32),
            bits=cfg['input_bits'], frac=cfg['input_frac'], from_int=True)
        y = XTensor(
            tensor=np.asarray(trace['y'], dtype=np.float32),
            bits=hw.Y_BITS, frac=acc_frac, from_int=True)

        if has_bias.get(name, True):
            b = XTensor(tensor=np.asarray(cfg['bias'], dtype=np.float32),
                        bits=hw.B_BITS, frac=cfg['bias_frac'], from_int=True)
        else:
            b = None

        is_last = ib == len(model.bundle_order) - 1
        if is_last and cfg['softmax']:
            pre_softmax = XTensor(
                tensor=np.asarray(model.pre_softmax, dtype=np.float32),
                bits=cfg['act_bits'], frac=model.softmax_frac, from_int=True)
            out = XTensor(
                tensor=np.asarray(model.softmax_out, dtype=np.float32),
                bits=None, float_only=True)
            softmax_frac = model.softmax_frac
            # sim.py:194 keeps a per-row maximum (shape (batch, 1)); config_fw.h
            # has ONE scalar per bundle and legacy uses a single global maximum
            # (xbundle.py:119). The hardware shares one maximum across the batch,
            # so collapse rather than pass an array.
            softmax_max_i = int(np.max(model.softmax_max_i))
        else:
            pre_softmax = None
            out = act_out
            softmax_frac = 0
            softmax_max_i = 0

        prev_ib = index_of[cfg['input']] if cfg['input'] is not None else None
        adapter = BrevitasBundle(
            ib=ib,
            core=_Core(w=w, x=x, y=y, b=b, act=act),
            softmax=bool(cfg['softmax']),
            out=out,
            pre_softmax=pre_softmax,
            prev_ib=prev_ib)

        adapter.softmax_frac = softmax_frac
        adapter.softmax_max_i = softmax_max_i

        adapters.append(adapter)
        BUNDLES.append(adapter)

    for adapter in adapters:
        if adapter.prev_ib is not None:
            adapters[adapter.prev_ib].next_ibs.add(adapter.ib)

    return adapters
