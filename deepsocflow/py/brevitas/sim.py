import json

import numpy as np

from deepsocflow.py.brevitas.lut import ActLut, CURVED_ACTIVATIONS, clip_to

# Activations whose act_quant is unsigned (Uint8ActPerTensorFixedPoint in ptq.py's
# ACT_MAP) - their quantized output range is [0, 2**bits-1], not the signed
# [-2**(bits-1), 2**(bits-1)-1] used everywhere else. export_graph_json now emits
# act_signed/input_signed per tensor directly (ptq.py), so this name-based table is
# only a fallback for JSONs exported before those fields existed - the primary
# mechanism is reading act_signed/input_signed straight from the JSON.
UNSIGNED_ACTIVATIONS = {'relu', 'sigmoid'}

# Activations this simulator executes as shift + clip, mirroring quant_lrelu
# (deepsocflow/c/runtime.h:153) - the closed form only works for functions that are
# piecewise-linear through the origin.
SUPPORTED_ACTIVATIONS = {'relu', 'identity'}

# Everything curved (silu/tanh/sigmoid/gelu/selu) is executed instead as a value
# LUT - still pure integer, still no multiplier, just a table load. See
# deepsocflow/py/brevitas/lut.py. Together these two sets are what this simulator
# can run; anything outside both is rejected in _build_topology.


def shift_round(n, s):
    """Round-to-nearest-even right shift on an int64 array. Mirrors
    deepsocflow/py/utils.py::shift_round and deepsocflow/c/runtime.h's shift_round
    macro exactly - same formula, just on numpy arrays instead of a scalar."""
    n = np.asarray(n, dtype=np.int64)
    if s <= 0:
        return n << (-s)
    half = np.int64(1) << (s - 1)
    return (n + half - (~(n >> s) & 1)) >> s


def conv2d_same_int(x_nhwc, w_hwio):
    """(N,H,W,CI) x (KH,KW,CI,CO) -> (N,H,W,CO), stride 1, 'same' padding, int64.

    The integer twin of rtl_export.py::_conv2d_same, and deliberately the same
    padding convention: pad_total = kernel - 1 with the extra pixel going to the
    bottom/right. That exporter accumulates in float32 for its golden per-pass
    sums; this one stays in int64 throughout, because it is the model the
    hardware is checked against and must not inherit a mantissa limit.

    Stride is not a parameter, and that is not an omission: the engine always
    convolves at stride 1: conv striding is the CPU dropping output pixels
    afterwards (runtime.h's CONV STRIDING block). forward() applies that drop.
    """
    x_nhwc = np.asarray(x_nhwc, dtype=np.int64)
    w_hwio = np.asarray(w_hwio, dtype=np.int64)
    N, H, W, CI = x_nhwc.shape
    KH, KW, w_ci, CO = w_hwio.shape
    assert w_ci == CI, f"weight input channels {w_ci} != input channels {CI}"

    pad_h, pad_w = KH - 1, KW - 1
    pad_top, pad_left = pad_h // 2, pad_w // 2
    xp = np.pad(x_nhwc, ((0, 0), (pad_top, pad_h - pad_top),
                         (pad_left, pad_w - pad_left), (0, 0)))
    out = np.zeros((N, H, W, CO), dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            out += np.einsum('nhwc,cd->nhwd', xp[:, kh:kh + H, kw:kw + W, :], w_hwio[kh, kw])
    return out


def c_div(a, b):
    """C integer division on int64 arrays: truncates toward zero.

    numpy's // floors instead, which differs for negatives - and div_round below
    depends on the C behaviour twice over, so this cannot be approximated.
    """
    a = np.asarray(a, dtype=np.int64)
    q = np.abs(a) // abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def div_round(a, b):
    """runtime.h's div_round macro, on int64 arrays.

    Transcribed rather than reimplemented, because its tie-break is not the
    obvious one: it deviates from round-half-away-from-zero on ~46% of inputs
    (always by at most 1, almost always on negatives - div_round(-4, 4) is 0, not
    -1). Average pooling's bit-exactness is exactly the claim that this quirk is
    reproduced, so it is pinned against the compiled macro itself by
    test_div_round_matches_c (deepsocflow/test/c/div_round_dump.c).
    """
    a = np.asarray(a, dtype=np.int64)
    correction = (~(b | c_div(a, b))) & 1
    return c_div(a + (b // 2) - correction, b)


def nchw_to_nhwc_flatten_perm(c, h, w):
    """Column permutation taking a dense weight from torch's flatten order to the
    engine's.

    torch's nn.Flatten runs over NCHW, so feature index = (c*H + h)*W + w - C
    slowest, W fastest. runtime.h's tile_write flattens NHWC instead:
    i_yc = (i_yh*yw + i_yw)*yc + i_yc, i.e. H slowest, C fastest. The two orders
    hold the same values in different positions, so a dense layer trained on
    torch's order needs its input columns reordered to read the engine's.

    Returns an array `perm` where perm[engine_index] == torch_index, so
    `weight[:, perm]` is the reordered weight. Getting this backwards produces a
    model that runs, trains, and is wrong in a way only a numeric comparison
    catches.
    """
    return np.arange(c * h * w).reshape(c, h, w).transpose(1, 2, 0).reshape(-1)


def maxpool2d_valid_int(x_nhwc, size, strides):
    """(N,H,W,C) -> (N,OH,OW,C), 'valid' max pooling, int64.

    Output size is dataflow.py:71's formula ((YH-PKH+PSH)//PSH) rather than a
    ceiling, matching torch's ceil_mode=False.

    Max rather than average is the whole point: max selects a value that already
    exists on the activation's grid, so there is nothing to requantize and
    nothing that has to agree with runtime.h's div_round.
    """
    x = np.asarray(x_nhwc, dtype=np.int64)
    (kh, kw), (sh, sw) = tuple(size), tuple(strides)
    n, h, w, c = x.shape
    oh, ow = (h - kh + sh) // sh, (w - kw + sw) // sw
    out = np.empty((n, oh, ow, c), dtype=np.int64)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = x[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :].max(axis=(1, 2))
    return out


def avgpool2d_valid_int(x_nhwc, size, strides):
    """(N,H,W,C) -> (N,OH,OW,C), 'valid' average pooling in the engine's integer
    arithmetic: sum the window, then div_round by the (constant) window size.

    Not a mean: div_round is runtime.h's macro, which deviates from ordinary
    rounding by up to 1 LSB on about half its inputs. Reproducing that quirk is
    the whole point - see div_round's docstring.
    """
    x = np.asarray(x_nhwc, dtype=np.int64)
    (kh, kw), (sh, sw) = tuple(size), tuple(strides)
    n, h, w, c = x.shape
    oh, ow = (h - kh + sh) // sh, (w - kw + sw) // sw
    count = kh * kw
    out = np.empty((n, oh, ow, c), dtype=np.int64)
    for i in range(oh):
        for j in range(ow):
            window = x[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            out[:, i, j, :] = div_round(window.sum(axis=(1, 2)), count)
    return out


def _apply_conv_stride(y_nhwc, stride, kernel_size):
    """Drops the output pixels a strided conv skips, from an un-strided conv sum.

    The start offset is dataflow.py:44-46's CSH_SHIFT/CSW_SHIFT verbatim, and it
    is not simply zero: under TF-style 'same' padding the first kept pixel sits
    at (K-1)//2 minus however much padding landed on the top/left, which for a
    3x3 stride-2 conv over an 8-wide axis is index 1, not index 0. Note torch's
    own Conv2d(stride=2, padding=1) anchors at index 0 instead - the two
    conventions differ by one pixel, so a float model that has to match this
    hardware must be built with the TF convention, not with torch's plain
    symmetric padding.
    """
    (sh, sw), (kh, kw) = tuple(stride), tuple(kernel_size)
    if (sh, sw) == (1, 1):
        return y_nhwc
    _, xh, xw, _ = y_nhwc.shape
    cyh, cyw = -(-xh // sh), -(-xw // sw)  # ceil division
    csh_shift = (kh - 1) // 2 - max((sh * (cyh - 1) + kh - xh) // 2, 0)
    csw_shift = (kw - 1) // 2 - max((sw * (cyw - 1) + kw - xw) // 2, 0)
    return y_nhwc[:, csh_shift::sh, csw_shift::sw, :]


def to_hwio(weight_int):
    """torch conv weight (CO,CI,KH,KW) -> engine layout (KH,KW,CI,CO).

    Named for the destination rather than the source because both the exporter
    (rtl_export.py:72) and dataflow.py's reorder helpers read weights in this
    layout; torch's is the odd one out.
    """
    return np.asarray(weight_int).transpose(2, 3, 1, 0)


class FixedPointModel:
    # Reads an already-quantized `quantized_model.export_graph_json()` graph (e.g.
    # deepsocflow/py/brevitas/model/xor_graph.json) and executes it in pure int64
    # arithmetic - matmul + bias-add + shift_round + relu/identity, mirroring
    # deepsocflow/c/runtime.h's quant_lrelu exactly. This is a bit-exact software
    # model of what the C firmware/RTL actually computes - unlike brevitas itself
    # (fake quantization: quantize+dequantize round-trip computed in float32), the
    # matmul/bias-add/shift_round/activation-clipping pipeline never touches a float
    # after construction. The one deliberate exception is the softmax convenience
    # output (forward()'s softmax_out): it's computed in float64 for readability
    # since softmax is monotonic and argmax(logits_int) already equals
    # argmax(softmax(logits)) without it - see the Usage note below.
    #
    # Usage:
    #   model = FixedPointModel(json_path)   # 1. build the graph topology
    #   model.load_int_weights(json_path)    # 2. pull the quantized int weights in
    #   x_int = model.quantize_input(x_float)
    #   logits_int = model.forward(x_int)    # pre-softmax; softmax is monotonic,
    #                                         # so argmax(logits_int) == argmax(softmax(logits))

    def __init__(self, json_path, lut_grid=None):
        """lut_grid: optional {bundle_name: (in_bits, in_frac)} choosing the index
        grid for that bundle's activation LUT. Left unset, a curved activation
        gets the variant-1a default of indexing at its own output grid - which
        suits silu but not the saturating functions (see ActLut.for_bundle)."""
        self.bundle_order = []
        self.bundles = {}
        self.lut_grid = dict(lut_grid or {})
        self._build_topology(json_path)

    def _build_topology(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        self.bundle_order = list(spec.keys())
        for name in self.bundle_order:
            cfg = spec[name]
            if cfg['type'] not in ('linear', 'conv'):
                raise ValueError(
                    f"FixedPointModel only supports type='linear' and type='conv' "
                    f"bundles (bundle '{name}' has type='{cfg['type']}')")
            activation = cfg['activation']
            if activation not in SUPPORTED_ACTIVATIONS and activation not in CURVED_ACTIVATIONS:
                raise ValueError(
                    f"FixedPointModel can't execute activation '{activation}' "
                    f"(bundle '{name}') in integer arithmetic - "
                    f"{sorted(SUPPORTED_ACTIVATIONS)} run as shift+clip and "
                    f"{sorted(CURVED_ACTIVATIONS)} run as a value LUT.")

            bias_cfg = cfg.get('bias')
            bias_frac = bias_cfg['frac'] if bias_cfg is not None else cfg['input_frac'] + cfg['weight']['frac']
            bias_bits = bias_cfg['bits'] if bias_cfg is not None else None

            act_signed = cfg.get('act_signed')
            if act_signed is None:
                act_signed = cfg['activation'] not in UNSIGNED_ACTIVATIONS

            # input_signed describes what THIS bundle expects to receive (not the
            # producing bundle's own act_signed) - the real exported JSON always
            # sets it now (ptq.py), so the True fallback only matters for older
            # JSONs that predate the field.
            input_signed = cfg.get('input_signed')
            if input_signed is None:
                input_signed = True

            # Curved activations get a table; relu/identity stay on the cheaper
            # shift+clip path in forward(). in_signed is always True regardless of
            # act_signed: the index is an accumulator, which is signed even when
            # the activation's output is not (sigmoid being the obvious case).
            lut = None
            if activation in CURVED_ACTIVATIONS:
                # Precedence: an explicit lut_grid wins; otherwise a variant-1b
                # model's exported act_in_* grid, which is the one that makes the
                # table bit-exact; otherwise the variant-1a default of indexing at
                # the output grid.
                if name in self.lut_grid:
                    in_bits, in_frac = self.lut_grid[name]
                else:
                    in_bits = cfg.get('act_in_bits')
                    in_frac = cfg.get('act_in_frac')
                lut = ActLut.for_bundle(
                    activation=activation,
                    act_bits=cfg['act_bits'], act_frac=cfg['act_frac'],
                    act_signed=act_signed,
                    in_bits=in_bits, in_frac=in_frac, in_signed=True)

            is_conv = cfg['type'] == 'conv'
            if is_conv:
                # in_features/out_features are kept populated for conv too, as
                # channel counts: load_int_weights sizes a missing bias from
                # out_features, and nothing downstream needs the distinction.
                in_features, out_features = cfg['in_channels'], cfg['out_channels']
                conv = dict(
                    kernel_size=tuple(cfg['kernel_size']),
                    stride=tuple(cfg['stride']),
                    padding=tuple(cfg['padding']),
                    input_shape=tuple(cfg['input_shape']),
                )
            else:
                in_features, out_features = cfg['in_features'], cfg['out_features']
                conv = None

            skip_from = cfg.get('skip_from')
            add_cfg = None
            if skip_from is not None:
                add_cfg = dict(
                    activation=cfg.get('add_activation', 'identity'),
                    act_bits=cfg['add_act_bits'],
                    act_frac=cfg['add_act_frac'],
                    act_signed=cfg.get('add_act_signed', True),
                )
                assert add_cfg['activation'] in SUPPORTED_ACTIVATIONS, (
                    f"bundle '{name}': add activation '{add_cfg['activation']}' is not "
                    f"executable - runtime.h:452 hardwires quant_lrelu for the residual "
                    f"slot and Bundle_t carries no aa_lut_idx, so a table cannot be used "
                    f"there")

            pool = cfg.get('pool')
            if pool is not None and pool['type'] not in ('max', 'avg'):
                raise ValueError(
                    f"bundle '{name}': pooling type '{pool['type']}' is not executable "
                    f"- only 'max' and 'avg' have integer implementations here")

            self.bundles[name] = dict(
                input=cfg.get('input'),
                type=cfg['type'],
                conv=conv,
                skip_from=skip_from,
                add=add_cfg,
                pool=pool,
                flatten=bool(cfg.get('flatten', False)),
                in_features=in_features,
                out_features=out_features,
                input_bits=cfg['input_bits'],
                input_frac=cfg['input_frac'],
                input_signed=input_signed,
                weight_bits=cfg['weight']['bits'],
                weight_frac=cfg['weight']['frac'],
                bias_bits=bias_bits,
                bias_frac=bias_frac,
                activation=cfg['activation'],
                act_bits=cfg['act_bits'],
                act_frac=cfg['act_frac'],
                act_signed=act_signed,
                lut=lut,
                softmax=cfg['softmax'],
                weight=None,  # populated by load_int_weights()
                bias=None,
            )

    def load_int_weights(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        for name in self.bundle_order:
            cfg = spec[name]
            bundle = self.bundles[name]
            bundle['weight'] = np.array(cfg['weight']['values'], dtype=np.int64)
            bias_cfg = cfg.get('bias')
            if bias_cfg is not None:
                bundle['bias'] = np.array(bias_cfg['values'], dtype=np.int64)
            else:
                bundle['bias'] = np.zeros(bundle['out_features'], dtype=np.int64)

    def quantize_input(self, x_float):
        """Quantizes a real-valued input using the first bundle's input scale,
        clipping to what its input_bits can represent, matching brevitas's own
        input quantizer (which clips out-of-range values instead of wrapping).
        Branches on the first bundle's own input_signed like forward()'s
        inter-bundle requant does, though in practice this is always signed:
        the first bundle's input_quant is always Int8ActPerTensorFixedPoint
        (ptq.py), never the unsigned variant."""
        first = self.bundles[self.bundle_order[0]]
        x_float = np.asarray(x_float, dtype=np.float64)
        if first['type'] == 'conv':
            # Callers hand in torch-convention NCHW; everything from here on -
            # forward()'s conv, the trace, and the exporter that reads it - is
            # NHWC, so this is the single place the two conventions meet.
            assert x_float.ndim == 4, (
                f"a conv first bundle expects a 4-D NCHW input, got shape {x_float.shape}")
            x_float = x_float.transpose(0, 2, 3, 1)
        x_int = np.rint(x_float * 2 ** first['input_frac'])
        bits = first['input_bits']
        if first['input_signed']:
            x_int = np.clip(x_int, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        else:
            x_int = np.clip(x_int, 0, 2 ** bits - 1)
        return x_int.astype(np.int64)

    def forward(self, x_int):
        if any(b['weight'] is None for b in self.bundles.values()):
            raise RuntimeError(
                "FixedPointModel.forward() called before load_int_weights() - "
                "weight/bias arrays are still None")

        self.outputs = {}
        self.trace = {}
        # {bundle name: (C, H, W)} for bundles that flatten, so the consuming
        # dense layer knows what shape its input columns came from.
        self._flatten_chw = {}
        prev_frac = {}
        x_int = np.asarray(x_int, dtype=np.int64)

        for name in self.bundle_order:
            bundle = self.bundles[name]

            if bundle['input'] is None:
                inp = x_int
                inp_frac = self.bundles[self.bundle_order[0]]['input_frac']
            else:
                inp = self.outputs[bundle['input']]
                inp_frac = prev_frac[bundle['input']]

            if inp_frac != bundle['input_frac']:
                inp = shift_round(inp, inp_frac - bundle['input_frac'])
                if bundle['input_signed']:
                    inp = np.clip(inp, -2 ** (bundle['input_bits'] - 1), 2 ** (bundle['input_bits'] - 1) - 1)
                else:
                    inp = np.clip(inp, 0, 2 ** bundle['input_bits'] - 1)

            acc_frac = bundle['input_frac'] + bundle['weight_frac']
            assert acc_frac == bundle['bias_frac'], (
                f"bundle '{name}': accumulator frac {acc_frac} != bias frac {bundle['bias_frac']}")

            if bundle['type'] == 'conv':
                # y stays UN-STRIDED on purpose: it is what the exporter reads as
                # core.y (rtl_export.py:74) and what it recomputes per pass with
                # _conv2d_same, which is stride-1 only. Striding is applied after,
                # on the way into the accumulator, mirroring runtime.h's order
                # (CONV STRIDING sits between the pass-sum and ADD BIAS).
                bundle['weight_engine'] = bundle['weight']
                y = conv2d_same_int(inp, to_hwio(bundle['weight']))
                acc_in = _apply_conv_stride(y, bundle['conv']['stride'],
                                            bundle['conv']['kernel_size'])
            else:
                weight = bundle['weight']
                src = bundle['input']
                if src is not None and src in self._flatten_chw:
                    # The producing bundle flattened in the engine's H,W,C order;
                    # this weight was trained against torch's C,H,W. Reorder the
                    # columns so the two index the same features.
                    weight = weight[:, nchw_to_nhwc_flatten_perm(*self._flatten_chw[src])]
                # The exporter must ship the SAME weight this model computed with,
                # so publish it rather than leaving adapter.py to re-derive the
                # permutation - two derivations of it is exactly how the engine
                # ends up multiplying by a differently-ordered matrix than the
                # reference. Written per forward() rather than in place, so
                # calling forward() twice cannot permute twice.
                bundle['weight_engine'] = weight
                y = inp @ weight.T                # bias-free conv-sum (matmul only)
                acc_in = y

            acc = acc_in + bundle['bias']         # int64, frac = acc_frac

            lut = bundle['lut']
            if lut is not None:
                # Shift the accumulator onto the table's index grid, clip, load.
                # index_shift() is where the power-of-two guard lives: the rescale
                # has to be a right shift, never a multiply.
                idx = shift_round(acc, lut.index_shift(acc_frac))
                idx = clip_to(idx, lut.in_bits, lut.in_signed)
                out = lut.lookup(idx)
            else:
                acc_for_shift = np.clip(acc, 0, None) if bundle['activation'] == 'relu' else acc
                out = shift_round(acc_for_shift, acc_frac - bundle['act_frac'])
                out = clip_to(out, bundle['act_bits'], bundle['act_signed'])

            # Pooling runs after the core activation, mirroring runtime.h's order
            # (CORE ACT -> residual -> POOLING). 'act' keeps the pre-pool tensor:
            # the exporter's per-pass golden sums are computed before pooling, so
            # a check that wants the activation output cannot read 'out' once a
            # pool is present.
            # Residual add sits between the core activation and pooling, matching
            # runtime.h (CORE ACT -> RESIDUAL ADD -> POOLING) and legacy
            # xbundle.py:92-105. The two operands are added raw, exactly as the C
            # does: Bundle_t has no add_val_shift/add_a_shift, so there is nothing
            # to align with - _build_topology already asserted the fracs match.
            if bundle['skip_from'] is not None:
                add = bundle['add']
                out = out + self.outputs[bundle['skip_from']]
                acc_for_shift = np.clip(out, 0, None) if add['activation'] == 'relu' else out
                out = shift_round(acc_for_shift, bundle['act_frac'] - add['act_frac'])
                out = clip_to(out, add['act_bits'], add['act_signed'])

            act_out = out
            if bundle['pool'] is not None:
                pool_fn = (maxpool2d_valid_int if bundle['pool']['type'] == 'max'
                           else avgpool2d_valid_int)
                out = pool_fn(out, bundle['pool']['size'], bundle['pool']['strides'])

            if bundle['flatten']:
                # NHWC reshaped to (N, -1) is already H,W,C order - the engine's.
                # The consuming dense layer's weight is reordered to match when
                # it is read below, rather than reordering the data here, because
                # the weight is reordered once and the data would be every batch.
                n = out.shape[0]
                self._flatten_chw[name] = (out.shape[3], out.shape[1], out.shape[2])
                out = out.reshape(n, -1)

            self.trace[name] = {'x': inp, 'y': y, 'acc': acc, 'act': act_out, 'out': out}
            self.outputs[name] = out
            prev_frac[name] = bundle['act_frac']

        last_name = self.bundle_order[-1]
        last_bundle = self.bundles[last_name]
        logits_int = self.outputs[last_name]

        if last_bundle['softmax']:
            self.pre_softmax = logits_int
            self.softmax_frac = last_bundle['act_frac']
            logits_float = logits_int.astype(np.float64) / 2 ** self.softmax_frac
            factor = 2 ** 17  # fixed-point scale used by the legacy hardware softmax
            row_max = logits_float.max(axis=-1, keepdims=True)
            self.softmax_max_i = (row_max * factor).astype(np.int64)
            exp = np.exp(logits_float - row_max)
            self.softmax_out = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
        else:
            self.pre_softmax = None
            self.softmax_frac = None
            self.softmax_max_i = None
            self.softmax_out = None

        return logits_int

    def print_graph(self):
        rows = []
        for name in self.bundle_order:
            b = self.bundles[name]
            source = b['input'] or 'x (model input)'
            rows.append((name, f"{b['in_features']}->{b['out_features']}", source,
                         b['activation'], f"{b['act_bits']}b/frac{b['act_frac']}"))

        header = ('bundle', 'shape', 'source', 'activation', 'output')
        widths = [max(len(r[c]) for r in rows + [header]) for c in range(5)]
        for row in [header] + rows:
            print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))
