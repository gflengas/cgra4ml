import json

import numpy as np

# Activations whose act_quant is unsigned (Uint8ActPerTensorFixedPoint in ptq.py's
# ACT_MAP) - their quantized output range is [0, 2**bits-1], not the signed
# [-2**(bits-1), 2**(bits-1)-1] used everywhere else. export_graph_json now emits
# act_signed/input_signed per tensor directly (ptq.py), so this name-based table is
# only a fallback for JSONs exported before those fields existed - the primary
# mechanism is reading act_signed/input_signed straight from the JSON.
UNSIGNED_ACTIVATIONS = {'relu', 'sigmoid'}

# Activations this simulator can execute in pure integer arithmetic today - anything
# not piecewise-linear through the origin (silu/tanh/gelu/selu/...) has no
# integer-exact implementation yet (see CLAUDE.md Known Issues).
SUPPORTED_ACTIVATIONS = {'relu', 'identity'}


def shift_round(n, s):
    """Round-to-nearest-even right shift on an int64 array. Mirrors
    deepsocflow/py/utils.py::shift_round and deepsocflow/c/runtime.h's shift_round
    macro exactly - same formula, just on numpy arrays instead of a scalar."""
    n = np.asarray(n, dtype=np.int64)
    if s <= 0:
        return n << (-s)
    half = np.int64(1) << (s - 1)
    return (n + half - (~(n >> s) & 1)) >> s


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

    def __init__(self, json_path):
        self.bundle_order = []
        self.bundles = {}
        self._build_topology(json_path)

    def _build_topology(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        self.bundle_order = list(spec.keys())
        for name in self.bundle_order:
            cfg = spec[name]
            if cfg['type'] != 'linear':
                raise ValueError(
                    f"FixedPointModel only supports type='linear' bundles right now "
                    f"(bundle '{name}' has type='{cfg['type']}')")
            if cfg['activation'] not in SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"FixedPointModel can't execute activation '{cfg['activation']}' "
                    f"(bundle '{name}') in integer arithmetic yet - only "
                    f"{sorted(SUPPORTED_ACTIVATIONS)} are piecewise-linear through the "
                    f"origin. See CLAUDE.md Known Issues.")

            bias_cfg = cfg.get('bias')
            bias_frac = bias_cfg['frac'] if bias_cfg is not None else cfg['input_frac'] + cfg['weight']['frac']

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

            self.bundles[name] = dict(
                input=cfg.get('input'),
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
                input_bits=cfg['input_bits'],
                input_frac=cfg['input_frac'],
                input_signed=input_signed,
                weight_frac=cfg['weight']['frac'],
                bias_frac=bias_frac,
                activation=cfg['activation'],
                act_bits=cfg['act_bits'],
                act_frac=cfg['act_frac'],
                act_signed=act_signed,
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
        x_int = np.rint(np.asarray(x_float, dtype=np.float64) * 2 ** first['input_frac'])
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

            y = inp @ bundle['weight'].T          # bias-free conv-sum (matmul only)
            acc = y + bundle['bias']              # int64, frac = acc_frac

            acc_for_shift = np.clip(acc, 0, None) if bundle['activation'] == 'relu' else acc
            out = shift_round(acc_for_shift, acc_frac - bundle['act_frac'])

            if bundle['act_signed']:
                out = np.clip(out, -2 ** (bundle['act_bits'] - 1), 2 ** (bundle['act_bits'] - 1) - 1)
            else:
                out = np.clip(out, 0, 2 ** bundle['act_bits'] - 1)

            self.trace[name] = {'x': inp, 'y': y, 'acc': acc, 'out': out}
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
