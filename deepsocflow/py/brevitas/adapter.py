"""Adapts brevitas FixedPointModel bundles onto the legacy XBundle attribute
surface, so the legacy engine-layout export (deepsocflow/py/xmodel.py) and RTL
verification path run over brevitas-produced numbers unchanged.

brevitas owns the numbers; the legacy backend owns the file format. This module
is the only seam between them."""
import math
from types import SimpleNamespace

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


def to_legacy_conv_weight(weight_int):
    """torch Conv2d weight (CO, CI, KH, KW) -> engine layout (KH, KW, CI, CO).

    Unlike the dense case there is no reshape downstream: export_bundle's conv
    branch (rtl_export.py:72) hands core.w.itensor straight to the reorder
    helpers, so it has to already be in the engine's layout. Shares its
    definition with sim.py::to_hwio, which needs the same permutation to run the
    integer convolution."""
    from deepsocflow.py.brevitas.sim import to_hwio
    return to_hwio(weight_int)


from deepsocflow.py.numeric import BUNDLES
from deepsocflow.py.brevitas.xtensor import XTensor


def pad_single_input_channel(w, x, is_conv):
    """Pads a one-channel layer out to two channels with zeros.

    A layer with CI == 1 gives the engine CM_0 == 1, which sets its C_CI counter
    to max_in == 0. counter.sv then holds `first` and `last` asserted together
    and permanently, so `is_cin_last` (axis_weight_rotator.sv:451) is stuck high;
    every beat in proc_engine's DELAY_MUL-deep pipeline claims to be the last
    channel, and the engine emits DELAY_MUL + 2 output banks and stops. With
    KH > 1 and a batch it miscomputes instead of stalling. Measured: CI == 1
    fails, CI == 2 onwards is fine. See CLAUDE.md.

    Padding with zeros is exact rather than approximate - the added channel
    contributes zero to every accumulation - so this changes what the engine is
    told about the layer without changing a single output value. It costs one
    extra input channel of storage on the first layer only, and needs no RTL
    change, so the existing bitstream stays valid.

    Returns (w, x) unchanged when there is nothing to pad.
    """
    ci_axis_w, ci_axis_x = (2, 3) if is_conv else (0, 1)
    if w.shape[ci_axis_w] != 1:
        return w, x

    pad_w = [(0, 0)] * w.ndim
    pad_w[ci_axis_w] = (0, 1)
    pad_x = [(0, 0)] * x.ndim
    pad_x[ci_axis_x] = (0, 1)
    return np.pad(w, pad_w), np.pad(x, pad_x)


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


class _Add:
    """Stands in for legacy XAdd (residual/skip add).

    The export path reads only `source_ib` and the three activation params
    (rtl_export.py:324, :328). `add_val_shift`/`add_a_shift` exist on the legacy
    object and are computed there, but Bundle_t has no field for them and nothing
    ever transports them to the firmware - the C adds the two operands raw
    (runtime.h:451). They are set to 0 here explicitly rather than omitted, so a
    reader can see that the omission is the hardware's and not an oversight; the
    matching frac requirement is enforced by check_hardware instead.
    """

    add_val_shift = 0
    add_a_shift = 0

    def __init__(self, source_ib, non_zero, plog_slope, shift_bits, out):
        self.source_ib = source_ib
        self.act = _Act(non_zero=non_zero, plog_slope=plog_slope,
                        shift_bits=shift_bits, out=out)
        self.out = out


class _Pool:
    """Stands in for legacy XPool.

    The engine reads only `type`, the three activation params, and
    pool_layer.pool_size/strides/padding (rtl_export.py:325,333-336 and
    dataflow.py:60-72); `x` is touched solely inside an assertion message
    (rtl_export.py:563), which is eagerly evaluated, so it has to exist.

    The activation is the identity: max pooling selects a value that is already
    on the activation's output grid, so there is nothing to rescale. non_zero=1
    with plog_slope=0 and shift_bits=0 is how quant_lrelu spells 'pass through'.
    """

    def __init__(self, pool_type, size, strides, x, padding='valid'):
        self.type = pool_type
        self.act = _Act(non_zero=1, plog_slope=0, shift_bits=0, out=x)
        self.pool_layer = SimpleNamespace(
            pool_size=tuple(size), strides=tuple(strides), padding=padding)
        self.x = x


class _Core:
    """Stands in for legacy XDense/XConvBN. type/strides/padding are what
    get_runtime_params reads off a core (dataflow.py:34-47).

    padding is always 'same': it is the only mode the engine implements
    (rtl_export.py::_conv2d_same), and ptq.py::_conv_geometry rejects anything
    else at quantization time rather than letting it reach here.
    """

    padding = 'same'

    # Read by config_fw.h's writer (xmodel.py:234). Legacy derives them in
    # XDense.call_int via out.add_val_shift(self.b), which the adapter's inert
    # call_int never runs. add_val_shift (utils.py:66-81) returns
    # (max(y.frac,b.frac)-y.frac, max(y.frac,b.frac)-b.frac); brevitas's single
    # quantization point per bundle makes bias_frac == acc_frac (sim.py:166
    # asserts it), so both shifts are identically zero.
    bias_val_shift = 0
    bias_b_shift = 0

    def __init__(self, w, x, y, b, act, type='dense', strides=(1, 1)):
        self.w, self.x, self.y, self.b, self.act = w, x, y, b, act
        # Instance-level, not class-level: a network mixing conv and dense
        # bundles (the flatten -> dense head) needs these to differ per bundle.
        self.type = type
        self.strides = tuple(strides)


class BrevitasBundle:
    """One brevitas bundle wearing legacy XBundle's attribute surface."""

    def __init__(self, ib, core, softmax, out, pre_softmax, prev_ib,
                 pool=None, flatten=False, add=None):
        self.ib = ib
        self.core = core
        self.pool = pool
        self.add = add
        # None-vs-truthy matters: xmodel.py:233 emits is_flatten/is_softmax with
        # `is not None`, while xbundle.py:144 and xmodel.py:221 test truthiness.
        # Legacy stores None when absent (xbundle.py:41,44), so `False` here would
        # make every bundle claim to be flattened and softmaxed.
        # Truthy-or-None, never False - see the note above.
        self.flatten = True if flatten else None
        self.softmax = True if softmax else None
        self.out = out
        self.pre_softmax = pre_softmax
        self.prev_ib = prev_ib
        # Lists, not sets: rtl_export.py frees an add buffer when
        # buf['out'][-1] == b.ib, i.e. it reads the LAST consumer by index. A set
        # is neither ordered nor indexable, and one bundle feeding several
        # consumers is the normal shape for a residual network. Appended in
        # ascending ib because bundles are built in ib order.
        self.next_ibs = []
        self.next_add_ibs = []

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

        # core.act.out is the activation's own output, which is what the engine
        # produced - NOT the bundle output. They differ whenever a pool follows,
        # so read the pre-pool tensor sim.py records as 'act' (it falls back to
        # 'out' on bundles with no pool, where the two are the same tensor).
        act_out = XTensor(
            tensor=np.asarray(trace.get('act', trace['out']), dtype=np.float32),
            bits=cfg['act_bits'], frac=cfg['act_frac'], from_int=True)

        # Pooling keeps the activation's grid (max selects an existing value), so
        # the bundle output carries the same bits/frac as the activation.
        bundle_out = XTensor(
            tensor=np.asarray(trace['out'], dtype=np.float32),
            bits=cfg['act_bits'], frac=cfg['act_frac'], from_int=True)

        add_cfg = cfg.get('add')
        skip_from = cfg.get('skip_from')
        if add_cfg is not None:
            add_nzero, add_plog = act_params(add_cfg['activation'])
            # Both operands are on cfg['act_frac'] (check_hardware enforces it), so
            # the sum is too - that is the grid this shift starts from.
            add = _Add(source_ib=index_of[skip_from],
                       non_zero=add_nzero, plog_slope=add_plog,
                       shift_bits=add_plog + cfg['act_frac'] - add_cfg['act_frac'],
                       out=bundle_out)
        else:
            add = None

        pool_cfg = cfg.get('pool')
        pool = _Pool(pool_type=pool_cfg['type'], size=pool_cfg['size'],
                     strides=pool_cfg['strides'], x=act_out,
                     padding=pool_cfg['padding']) if pool_cfg else None

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

        is_conv = cfg['type'] == 'conv'
        to_legacy_weight = to_legacy_conv_weight if is_conv else to_legacy_dense_weight

        w_arr = to_legacy_weight(cfg['weight_engine']).astype(np.float32)
        x_arr = np.asarray(trace['x'], dtype=np.float32)
        # Only the engine's view is padded; sim.py's numbers are untouched, and
        # the padded channel is zero, so both still describe the same layer.
        w_arr, x_arr = pad_single_input_channel(w_arr, x_arr, is_conv)

        w = XTensor(tensor=w_arr, bits=hw.K_BITS, frac=cfg['weight_frac'], from_int=True)
        x = XTensor(tensor=x_arr, bits=cfg['input_bits'], frac=cfg['input_frac'],
                    from_int=True)
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
            out = bundle_out
            softmax_frac = 0
            softmax_max_i = 0

        prev_ib = index_of[cfg['input']] if cfg['input'] is not None else None
        adapter = BrevitasBundle(
            ib=ib,
            core=_Core(w=w, x=x, y=y, b=b, act=act,
                       type='conv' if is_conv else 'dense',
                       strides=cfg['conv']['stride'] if is_conv else (1, 1)),
            softmax=bool(cfg['softmax']),
            out=out,
            pre_softmax=pre_softmax,
            prev_ib=prev_ib,
            pool=pool,
            flatten=bool(cfg.get('flatten', False)),
            add=add)

        adapter.softmax_frac = softmax_frac
        adapter.softmax_max_i = softmax_max_i

        adapters.append(adapter)
        BUNDLES.append(adapter)

    for adapter in adapters:
        if adapter.prev_ib is not None:
            adapters[adapter.prev_ib].next_ibs.append(adapter.ib)
        # Ascending by construction (adapters are built in ib order), which is what
        # rtl_export.py's buf['out'][-1] free step relies on.
        if adapter.add is not None:
            adapters[adapter.add.source_ib].next_add_ibs.append(adapter.ib)

    return adapters
