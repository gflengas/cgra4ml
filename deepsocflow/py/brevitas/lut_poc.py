"""Measurement PoC for value-LUT activations (deepsocflow/py/brevitas/lut.py).

Answers one question with numbers rather than argument: how fine does the table's
index grid have to be before the LUT reproduces the exact quantized activation
bit-for-bit, and what does that cost in bytes?

Two experiments:

  1. `sweep_index_grid` - synthetic. For each curved activation, walk the index
     grid (in_bits x in_frac) and report the mismatch against
     `exact_activation` over a dense sweep of accumulators. Also reports the
     strict variant-1a point (index grid == output grid) for comparison, since
     that is the configuration that needs no new quantization point at all.

  2. `real_model` - end to end. Trains the XOR model with SiLU instead of
     LeakyReLU, quantizes it through ptq.py, exports the graph JSON, and runs
     FixedPointModel over it - checking the integer LUT pipeline against
     brevitas's own fake-quantized forward pass.

Run:  python -m deepsocflow.py.brevitas.lut_poc
      python -m deepsocflow.py.brevitas.lut_poc --sweep-only   (no torch needed)

Note: experiment 1 needs only numpy. Experiment 2 needs torch + brevitas.
"""

import argparse
import contextlib
import io
import math
import os
import sys
import tempfile

import numpy as np

from deepsocflow.py.brevitas.lut import ActLut, exact_activation, lut_activation, mismatch

# Output grids these activations get in practice, from ptq.py's quantizer choices:
# Int8ActPerTensorFixedPoint for the signed ones, Uint8 narrowed to bits-1 for
# sigmoid. act_frac is what calibration would pick for each output range.
#   silu/gelu/selu  ->  unbounded above, so the integer part dominates
#   tanh/sigmoid    ->  bounded in [-1,1] / [0,1], so nearly all bits are fractional
OUTPUT_GRIDS = {
    'silu':    dict(act_bits=8, act_frac=4, act_signed=True),
    'gelu':    dict(act_bits=8, act_frac=4, act_signed=True),
    'selu':    dict(act_bits=8, act_frac=5, act_signed=True),
    'tanh':    dict(act_bits=8, act_frac=7, act_signed=True),
    'sigmoid': dict(act_bits=7, act_frac=7, act_signed=False),
}

# The real-valued span each activation still varies over. Beyond this it has
# saturated (or gone linear) and extra table range buys nothing.
DOMAIN = {'silu': 8.0, 'gelu': 8.0, 'selu': 8.0, 'tanh': 4.0, 'sigmoid': 8.0}

ACC_FRAC = 13  # input_frac 7 + weight_frac 6, the XOR model's actual accumulator grid


def _acc_sweep(activation, acc_frac=ACC_FRAC, points=200_001):
    """Accumulators spanning the activation's live domain, at accumulator
    resolution. Uniform rather than model-derived on purpose: it weights the
    curved region as heavily as the saturated tails, so it is the pessimistic
    reading."""
    half = DOMAIN[activation]
    lo, hi = int(-half * 2 ** acc_frac), int(half * 2 ** acc_frac)
    return np.linspace(lo, hi, points).astype(np.int64)


def sweep_index_grid(activations=None, acc_frac=ACC_FRAC, max_bits=14):
    """For each activation and index width, find the in_frac that minimises
    mismatch, and report it. Searching in_frac rather than deriving it avoids
    baking in a hand-picked domain assumption."""
    activations = activations or list(OUTPUT_GRIDS)
    results = {}

    for activation in activations:
        grid = OUTPUT_GRIDS[activation]
        acc = _acc_sweep(activation, acc_frac)
        rows = []

        # strict variant 1a: index grid == output grid, no free parameters
        lut_1a = ActLut.for_bundle(activation, **grid)
        frac_1a, worst_1a = mismatch(acc, acc_frac, lut_1a)
        rows.append(dict(label='1a (index = output grid)', in_bits=lut_1a.in_bits,
                         in_frac=lut_1a.in_frac, mismatch=frac_1a, worst=worst_1a,
                         nbytes=lut_1a.nbytes, span=lut_1a.input_range))

        for in_bits in range(6, max_bits + 1):
            best = None
            for in_frac in range(0, min(in_bits, acc_frac) + 1):
                lut = ActLut.for_bundle(activation, in_bits=in_bits, in_frac=in_frac,
                                        **grid)
                frac, worst = mismatch(acc, acc_frac, lut)
                if best is None or (frac, worst) < (best['mismatch'], best['worst']):
                    best = dict(label=f'{in_bits}-bit index', in_bits=in_bits,
                                in_frac=in_frac, mismatch=frac, worst=worst,
                                nbytes=lut.nbytes, span=lut.input_range)
            rows.append(best)

        results[activation] = rows
    return results


def print_sweep(results):
    for activation, rows in results.items():
        grid = OUTPUT_GRIDS[activation]
        print(f"\n{activation.upper()}  "
              f"output {grid['act_bits']}b/frac{grid['act_frac']} "
              f"({'signed' if grid['act_signed'] else 'unsigned'}), "
              f"accumulator frac{ACC_FRAC}, live domain +/-{DOMAIN[activation]:g}")
        print(f"  {'config':<26} {'in_frac':>7} {'covers':>16} "
              f"{'mismatch':>9} {'worst':>6} {'bytes':>7}")
        print("  " + "-" * 76)
        for r in rows:
            lo, hi = r['span']
            exact = "  <- bit-exact" if r['mismatch'] == 0.0 else ""
            print(f"  {r['label']:<26} {r['in_frac']:>7} "
                  f"{f'[{lo:.3g},{hi:.3g}]':>16} "
                  f"{r['mismatch'] * 100:>8.3f}% {r['worst']:>6} {r['nbytes']:>7}{exact}")


def first_exact(rows):
    """The cheapest row in a sweep that reached zero mismatch, if any."""
    exact = [r for r in rows if r['mismatch'] == 0.0]
    return min(exact, key=lambda r: r['nbytes']) if exact else None


def summarise(results):
    print("\n\n" + "=" * 84)
    print("SUMMARY - cheapest bit-exact table per activation")
    print("=" * 84)
    print(f"  {'activation':<10} {'1a mismatch':>12} {'bit-exact at':>14} "
          f"{'in_frac':>8} {'bytes':>7}")
    print("  " + "-" * 76)
    for activation, rows in results.items():
        one_a = rows[0]
        best = first_exact(rows)
        if best is None:
            verdict = f"{'none <= 14b':>14} {'-':>8} {'-':>7}"
        else:
            verdict = f"{str(best['in_bits']) + '-bit':>14} {best['in_frac']:>8} {best['nbytes']:>7}"
        print(f"  {activation:<10} {one_a['mismatch'] * 100:>11.3f}% {verdict}")


# ---- experiment 2: real model ----

def _train_xor_silu(seed=30, epochs=4000, activation=None, quiet=False):
    """Train the XOR model and return (net, X, Y).

    `activation` defaults to SiLU - the activation xor.py originally had, before
    it was swapped for LeakyReLU precisely because nothing downstream could
    execute it. Any curved activation can be substituted to check that a result
    is not an accident of one function's shape."""
    import torch
    import torch.nn as nn

    act_cls = activation or nn.SiLU

    class XORCurved(nn.Module):
        """Same shape as deepsocflow/py/brevitas/xor.py's XOR."""

        def __init__(self):
            super().__init__()
            self.hidden_1 = nn.Linear(2, 8, bias=True)
            self.act_1 = act_cls()
            self.hidden_2 = nn.Linear(8, 8, bias=True)
            self.act_2 = act_cls()
            self.output = nn.Linear(8, 2, bias=True)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.act_1(self.hidden_1(x))
            x = self.act_2(self.hidden_2(x))
            return self.softmax(self.output(x))

    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = torch.tensor([0, 1, 1, 0])

    torch.manual_seed(seed)
    net = XORCurved()
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    loss_fn = nn.NLLLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(torch.log(net(X) + 1e-12), Y)
        loss.backward()
        opt.step()

    float_pred = net(X).argmax(-1).tolist()
    if not quiet:
        print(f"\n  float model predictions: {float_pred}  (XOR is [0, 1, 1, 0])")
        if float_pred != [0, 1, 1, 0]:
            print("  !! float model did not converge to XOR - try another seed")
    return net, X, Y


def real_model(seed=30, epochs=4000):
    """Train XOR with SiLU, quantize, export, and run the integer LUT pipeline
    against brevitas's own forward pass."""
    import torch

    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.sim import FixedPointModel

    net, X, _ = _train_xor_silu(seed, epochs)

    qm = quantized_model(net, weight_bits=8, bias_bits=16)
    qm.quantization(X)
    import tempfile
    import os
    json_path = os.path.join(tempfile.mkdtemp(), 'xor_silu_graph.json')
    qm.export_graph_json(X, json_path)
    print(f"  graph JSON: {json_path}")

    brevitas_out = qm(X)
    brevitas_pred = brevitas_out.argmax(-1).tolist()

    brevitas_probs = brevitas_out.detach().numpy()

    # Grids to compare. The last one indexes at full accumulator resolution and
    # is sized from the accumulators the model actually produces - the only
    # configuration that can be bit-exact (see sweep_index_grid), included to
    # show what exactness really costs.
    exact_grid = exact_lut_grid(json_path, X.numpy())

    configs = [
        ("1a (index = output grid)", None),
        ("10-bit index @ frac 6", {f'bundle{i}': (10, 6) for i in range(3)}),
        ("12-bit index @ frac 8", {f'bundle{i}': (12, 8) for i in range(3)}),
        ("index at full acc resolution", exact_grid),
    ]

    print(f"\n  {'config':<30} {'pred':<14} {'act mismatch':>13} {'worst':>6} "
          f"{'softmax err':>12} {'bytes':>8}")
    print("  " + "-" * 88)
    for label, lut_grid in configs:
        model = FixedPointModel(json_path, lut_grid=lut_grid)
        model.load_int_weights(json_path)
        x_int = model.quantize_input(X.numpy())
        logits = model.forward(x_int)
        pred = np.asarray(logits).argmax(-1).tolist()

        # Per-bundle: does the table reproduce the activation the accumulator
        # actually deserves? Measured on the real accumulators, not a sweep.
        differing = total = worst = 0
        for name in model.bundle_order:
            bundle = model.bundles[name]
            lut = bundle['lut']
            if lut is None:
                continue
            acc = model.trace[name]['acc']
            acc_frac = bundle['input_frac'] + bundle['weight_frac']
            ref = exact_activation(acc, acc_frac, lut.activation, lut.out_bits,
                                   lut.out_frac, lut.out_signed)
            got = lut_activation(acc, acc_frac, lut)
            diff = np.abs(got - ref)
            differing += int((diff != 0).sum())
            total += diff.size
            worst = max(worst, int(diff.max()))

        softmax_err = float(np.abs(model.softmax_out - brevitas_probs).max())
        luts = [b['lut'] for b in model.bundles.values() if b['lut'] is not None]
        total_bytes = sum(l.nbytes for l in luts)
        ok = "OK" if pred == [0, 1, 1, 0] else "BAD"
        rate = differing / total * 100 if total else 0.0
        print(f"  [{ok:<3}] {label:<28} {str(pred):<14} {rate:>12.2f}% {worst:>6} "
              f"{softmax_err:>12.2e} {total_bytes:>8}")

    print(f"\n  brevitas fake-quant reference predictions: {brevitas_pred}")
    print("  'act mismatch' is per-activation-value disagreement with the exact "
          "quantized\n  activation; 'softmax err' is max abs error of the final "
          "probabilities vs brevitas.")
    return json_path


def exact_lut_grid(json_path, x_float):
    """The per-bundle index grid that makes every LUT in the model bit-exact.

    Two conditions have to hold together, and both are measured here rather than
    assumed:

      in_frac == acc_frac   so no accumulator detail is lost to bucketing
      in_bits  wide enough  that no real accumulator saturates at a table edge

    The second needs the accumulators the model actually produces, so this runs
    a probe forward pass to find their range. Sizing from a nominal domain
    instead leaves a few saturating values mismatched - which is exactly what a
    fixed `acc_frac + 4` guess does on this model."""
    from deepsocflow.py.brevitas.sim import FixedPointModel

    probe = FixedPointModel(json_path)
    probe.load_int_weights(json_path)
    probe.forward(probe.quantize_input(x_float))

    grid = {}
    for name in probe.bundle_order:
        bundle = probe.bundles[name]
        if bundle['lut'] is None:
            continue
        acc_frac = bundle['input_frac'] + bundle['weight_frac']
        acc = probe.trace[name]['acc']
        peak = max(abs(int(acc.min())), abs(int(acc.max()))) / 2.0 ** acc_frac
        int_bits = max(1, int(math.ceil(math.log2(peak))) if peak > 0 else 1)
        grid[name] = (acc_frac + int_bits + 1, acc_frac)
    return grid


def real_model_1b(seed=30, epochs=4000, act_input_bits=8):
    """Variant 1b on a real model: quantize each SiLU's input too, then check the
    256 B table against brevitas's own forward pass.

    This is the claim `variant_1b` demonstrates synthetically, done for real -
    including the part the synthetic version cannot show, which is what
    calibration actually picks for the activation input scale and what the extra
    quantization point costs the model's decision margin."""
    import torch

    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.sim import FixedPointModel

    net, X, _ = _train_xor_silu(seed, epochs)

    rows = []
    for label, bits in (("1a (no input_quant)", None),
                        (f"1b (input_quant {act_input_bits}b)", act_input_bits)):
        qm = quantized_model(net, weight_bits=8, bias_bits=16, act_input_bits=bits)
        qm.quantization(X)
        import os
        import tempfile
        json_path = os.path.join(tempfile.mkdtemp(), 'g.json')
        qm.export_graph_json(X, json_path)

        # Capture brevitas's own activation output per bundle. This is the only
        # honest reference: comparing against exact_activation() on the raw
        # accumulator silently assumes the 1a definition of "exact", which is
        # precisely what 1b changes.
        captured = {}

        def _hook(idx):
            def fn(_module, _inp, out):
                captured[idx] = out
            return fn

        handles = [b.core.act.register_forward_hook(_hook(i))
                   for i, b in enumerate(qm.bundles)]
        with torch.no_grad():
            probs = qm(X).detach().numpy()
        for h in handles:
            h.remove()

        model = FixedPointModel(json_path)
        model.load_int_weights(json_path)
        model.forward(model.quantize_input(X.numpy()))

        differing = total = worst = 0
        grids = []
        for idx, name in enumerate(model.bundle_order):
            bundle = model.bundles[name]
            lut = bundle['lut']
            if lut is None:
                continue
            # brevitas's activation output, converted from its fake-quantized
            # float back to the integer level the hardware would hold
            qt = captured[idx]
            ref = np.rint(qt.value.detach().numpy() / qt.scale.item()).astype(np.int64)
            diff = np.abs(model.trace[name]['out'] - ref)
            differing += int((diff != 0).sum())
            total += diff.size
            worst = max(worst, int(diff.max()))
            grids.append(f"{lut.in_bits}b/frac{lut.in_frac}")

        # decision margin: how far the winning class sits above the other, the
        # thing an extra quantization point could actually erode
        margin = float((np.sort(probs, axis=-1)[:, -1] - np.sort(probs, axis=-1)[:, -2]).min())
        rows.append(dict(
            label=label, grids=grids, mismatch=differing / total * 100 if total else 0.0,
            worst=worst, margin=margin,
            bytes=sum(b['lut'].nbytes for b in model.bundles.values() if b['lut']),
            pred=np.asarray(model.forward(model.quantize_input(X.numpy()))).argmax(-1).tolist()))

    print(f"\n  {'config':<26} {'index grid':<20} {'vs brevitas':>12} {'worst':>6} "
          f"{'margin':>8} {'bytes':>7}")
    print("  " + "-" * 84)
    for r in rows:
        ok = "OK" if r['pred'] == [0, 1, 1, 0] else "BAD"
        print(f"  [{ok:<3}] {r['label']:<24} {', '.join(r['grids']):<20} "
              f"{r['mismatch']:>11.2f}% {r['worst']:>6} {r['margin']:>8.4f} {r['bytes']:>7}")
    print("\n  'vs brevitas' is the LUT pipeline's disagreement with brevitas's own")
    print("  fake-quantized activation output - the number that has to be 0 for the")
    print("  golden reference to stay bit-exact. 'margin' is the smallest gap between")
    print("  the winning and runner-up class over the 4 rows (higher is safer).")
    return rows


CURVED = ('silu', 'tanh', 'gelu', 'sigmoid', 'selu')


def _measure(net, X, activation_name, act_input_bits):
    """Quantize a trained net at a given input width and measure both things
    that matter: whether the table still reproduces brevitas exactly, and what
    the quantization did to the model itself."""
    import torch

    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.sim import FixedPointModel

    with contextlib.redirect_stdout(io.StringIO()):
        qm = quantized_model(net, weight_bits=8, bias_bits=16,
                             act_input_bits=act_input_bits)
        qm.quantization(X)
    json_path = os.path.join(tempfile.mkdtemp(), 'g.json')
    qm.export_graph_json(X, json_path)

    captured = {}

    def _hook(idx):
        def fn(_m, _i, out):
            captured[idx] = out
        return fn

    handles = [b.core.act.register_forward_hook(_hook(i)) for i, b in enumerate(qm.bundles)]
    with torch.no_grad():
        probs = qm(X).detach().numpy()
    for h in handles:
        h.remove()

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    model.forward(model.quantize_input(X.numpy()))

    differing = total = worst = 0
    nbytes = 0
    for idx, name in enumerate(model.bundle_order):
        lut = model.bundles[name]['lut']
        if lut is None:
            continue
        qt = captured[idx]
        ref = np.rint(qt.value.detach().numpy() / qt.scale.item()).astype(np.int64)
        diff = np.abs(model.trace[name]['out'] - ref)
        differing += int((diff != 0).sum())
        total += diff.size
        worst = max(worst, int(diff.max()))
        nbytes += lut.nbytes

    ordered = np.sort(probs, axis=-1)
    return dict(
        mismatch=differing / total * 100 if total else 0.0,
        worst=worst, nbytes=nbytes,
        pred=model.softmax_out.argmax(-1).tolist() if model.softmax_out is not None
        else np.asarray(model.forward(model.quantize_input(X.numpy()))).argmax(-1).tolist(),
        margin=float((ordered[:, -1] - ordered[:, -2]).min()),
    )


def all_activations_1b(act_input_bits=(4, 5, 6, 7, 8, 10), activations=CURVED,
                       seed=30, epochs=4000):
    """1a vs 1b across every curved activation, on real trained models, over a
    wide range of input widths.

    Two independent questions are answered in one table:

      does the table still match brevitas   -> should stay 0% at every width,
                                               because exactness comes from the
                                               table sharing brevitas's grid,
                                               not from that grid being wide
      what does narrowing cost the model    -> shows up as a shrinking decision
                                               margin, and eventually a wrong
                                               prediction

    Training is done once per activation and reused across widths, since
    act_input_bits only affects quantization."""
    import torch.nn as nn

    act_types = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU,
                 'sigmoid': nn.Sigmoid, 'selu': nn.SELU}

    width = 15
    head = f"  {'activation':<9} {'1a @256B':>{width}} " + " ".join(
        f"{'1b @' + str(b) + 'b':>{width}}" for b in act_input_bits)
    print("\n  MISMATCH vs brevitas  (0% = the golden reference stays bit-exact)")
    print(head)
    print("  " + "-" * (len(head) - 2))

    quality = {}
    for name in activations:
        net, X, _ = _train_xor_silu(seed, epochs, activation=act_types[name], quiet=True)
        cells = []

        r = _measure(net, X, name, None)
        cells.append(f"{r['mismatch']:>6.1f}% /{r['worst']:>3}LSB")
        quality[name] = [('1a', r)]

        for bits in act_input_bits:
            r = _measure(net, X, name, bits)
            flag = "" if r['mismatch'] == 0 else " !!"
            cells.append(f"{r['mismatch']:>6.2f}% {r['nbytes']:>4}B{flag}")
            quality[name].append((f'{bits}b', r))

        print(f"  {name:<9} " + " ".join(f"{c:>{width}}" for c in cells))

    print("\n\n  MODEL QUALITY  (decision margin; * = a prediction went wrong)")
    head2 = f"  {'activation':<9} {'1a':>{width}} " + " ".join(
        f"{str(b) + 'b':>{width}}" for b in act_input_bits)
    print(head2)
    print("  " + "-" * (len(head2) - 2))
    for name in activations:
        cells = []
        for _, r in quality[name]:
            bad = "*" if r['pred'] != [0, 1, 1, 0] else " "
            cells.append(f"{r['margin']:>8.4f}{bad}")
        print(f"  {name:<9} " + " ".join(f"{c:>{width}}" for c in cells))

    print("\n  Left table: table-vs-model agreement. Right table: model-vs-task quality.")
    print("  They move independently - that separation is the point of variant 1b.")
    return quality


def variant_1b(acc_frac=ACC_FRAC, in_bits=8):
    """Why variant 1b reaches bit-exactness at 256 B when 1a cannot at 128 KB.

    1a and 1b are not two table layouts - they are two different definitions of
    "exact", and that is the whole finding:

      1a  brevitas applies the activation to the full-precision accumulator, so
          the reference carries acc_frac bits of input detail. A table can only
          match it by indexing at acc_frac too, which is why exactness costs
          tens of kilobytes.

      1b  an `input_quant` on the activation makes brevitas quantize the
          accumulator *first*. The reference now carries only in_bits of input
          detail, and a table on that same grid reproduces it exactly by
          construction - at 2**in_bits entries.

    So 1b does not approximate better; it moves the approximation from the
    deployment step (silent, discovered at RTL-diff time) into the model itself
    (visible during calibration, and trainable through). This function measures
    both sides of that claim: the table is exact against the 1b reference, and
    the 1b reference itself differs from the full-precision one by this much."""
    print(f"\n  {'activation':<10} {'vs 1b reference':>16} {'1b vs full-precision':>21} "
          f"{'worst':>6} {'bytes':>7}")
    print("  " + "-" * 68)
    for activation, grid in OUTPUT_GRIDS.items():
        half = DOMAIN[activation]
        in_frac = in_bits - 1 - int(math.ceil(math.log2(half)))
        lut = ActLut.for_bundle(activation, in_bits=in_bits, in_frac=in_frac, **grid)
        acc = _acc_sweep(activation, acc_frac)

        # what input_quant would produce: the accumulator on the table's grid
        idx = lut_index(acc, acc_frac, lut)
        # 1b reference - activation applied to the already-quantized input
        ref_1b = exact_activation(idx, lut.in_frac, activation, lut.out_bits,
                                  lut.out_frac, lut.out_signed)
        got = lut.lookup(idx)
        exact_vs_1b = float((got != ref_1b).mean())

        # what that input quantization costs the model, against 1a's reference
        ref_full = exact_activation(acc, acc_frac, activation, lut.out_bits,
                                    lut.out_frac, lut.out_signed)
        cost = float((ref_1b != ref_full).mean())
        worst = int(np.abs(ref_1b - ref_full).max())

        print(f"  {activation:<10} {exact_vs_1b * 100:>15.4f}% {cost * 100:>20.2f}% "
              f"{worst:>6} {lut.nbytes:>7}")
    print("\n  'vs 1b reference' is the table's own error - zero by construction.")
    print("  '1b vs full-precision' is what the input quantization costs the model,")
    print("  which calibration sees and QAT can train against.")


def lut_index(acc, acc_frac, lut):
    """The index a given accumulator lands on - the shift+clip half of the
    pipeline, separated out so measurement code can inspect it."""
    from deepsocflow.py.brevitas.lut import clip_to
    from deepsocflow.py.brevitas.sim import shift_round
    return clip_to(shift_round(acc, lut.index_shift(acc_frac)), lut.in_bits, lut.in_signed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep-only', action='store_true',
                    help='skip the torch/brevitas experiment')
    ap.add_argument('--max-bits', type=int, default=14)
    args = ap.parse_args()

    print("=" * 84)
    print("EXPERIMENT 1 - how fine must the index grid be for a bit-exact LUT?")
    print("=" * 84)
    results = sweep_index_grid(max_bits=args.max_bits)
    print_sweep(results)
    summarise(results)

    print("\n\n" + "=" * 84)
    print("EXPERIMENT 3 - variant 1b: quantize the activation's input, and 256 B is exact")
    print("=" * 84)
    variant_1b()

    if not args.sweep_only:
        print("\n\n" + "=" * 84)
        print("EXPERIMENT 2 - XOR with SiLU, end to end through the integer pipeline")
        print("=" * 84)
        try:
            real_model()
            print("\n\n" + "=" * 84)
            print("EXPERIMENT 4 - variant 1b on the real model, checked against brevitas")
            print("=" * 84)
            real_model_1b()

            print("\n\n" + "=" * 84)
            print("EXPERIMENT 5 - every curved activation, every input width")
            print("=" * 84)
            all_activations_1b()
        except ImportError as e:
            print(f"  skipped: {e}")
            return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
