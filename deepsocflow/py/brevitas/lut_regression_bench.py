"""Benchmark: how many bits does a curved activation's input quantization need?

Answers the question the XOR model could not.

XOR hides activation error behind an argmax and near-saturated probabilities - a
prediction is right or wrong, and 1-LSB noise never flips it. A regression target
has no such cushion: every LSB of activation error lands directly in the RMSE.

Dataset: UCI Airfoil Self-Noise (1503 x 5 -> scaled sound pressure level, dB).
Model:   5 -> 32 -> 32 -> 1, SiLU (or any curved activation), no softmax.

Reports, per configuration:
  RMSE (dB)        of the integer LUT pipeline on held-out data
  vs brevitas      per-activation-value disagreement (0% = golden ref stays exact)
  table bytes      total across the model
"""

import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from deepsocflow.py.brevitas.ptq import quantized_model
from deepsocflow.py.brevitas.sim import FixedPointModel

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/"
       "airfoil_self_noise.dat")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "airfoil_self_noise.dat")
SEED = 0


def fetch():
    """Download the dataset on first use. Not checked in - it is third-party data
    and this is the only thing in the repo that needs it."""
    if not os.path.exists(DATA):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, DATA)
    return DATA


def load():
    """Standardised train/test split. Standardising the target matters here:
    the pipeline's input/activation quantisers are calibrated on real ranges, and
    a target centred near 125 dB with std 7 would put the whole network in a
    regime where the integer datapath spends its bits on the offset."""
    raw = np.loadtxt(fetch())
    X, y = raw[:, :5], raw[:, 5]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = perm[:n_tr], perm[n_tr:]

    mu, sd = X[tr].mean(0), X[tr].std(0)
    ymu, ysd = y[tr].mean(), y[tr].std()
    Xs = ((X - mu) / sd).astype(np.float32)
    ys = ((y - ymu) / ysd).astype(np.float32)
    return (torch.tensor(Xs[tr]), torch.tensor(ys[tr]),
            torch.tensor(Xs[te]), torch.tensor(ys[te]), float(ysd))


class MLP(nn.Module):
    """Attributes are registered in execution order (hidden_i, act_i, ...), which
    is what ptq.py's quantized_model walks to pair each compute layer with the
    activation that follows it - see the 2026-08-10 note about a shared activation
    attribute breaking that pairing."""

    def __init__(self, act=nn.SiLU, hidden=(32, 32), in_features=5):
        super().__init__()
        prev = in_features
        for i, w in enumerate(hidden, start=1):
            setattr(self, f'hidden_{i}', nn.Linear(prev, w, bias=True))
            setattr(self, f'act_{i}', act())
            prev = w
        self.out = nn.Linear(prev, 1, bias=True)
        self.n_hidden = len(hidden)

    def forward(self, x):
        for i in range(1, self.n_hidden + 1):
            x = getattr(self, f'act_{i}')(getattr(self, f'hidden_{i}')(x))
        return self.out(x)


def train(act=nn.SiLU, epochs=2000, hidden=(32, 32), lr=3e-3):
    Xtr, ytr, Xte, yte, ysd = load()
    torch.manual_seed(SEED)
    net = MLP(act, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(net(Xtr).squeeze(-1), ytr).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        rmse = float(torch.sqrt(nn.functional.mse_loss(
            net(Xte).squeeze(-1), yte))) * ysd
    return net, Xtr, Xte, yte, ysd, rmse


def evaluate(net, Xtr, Xte, yte, ysd, act_input_bits, tmpdir):
    """Quantize at a given input width, then run the integer LUT pipeline over the
    held-out set and compare against brevitas's own activation outputs."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        qm = quantized_model(net, weight_bits=8, bias_bits=16,
                             act_input_bits=act_input_bits)
        qm.quantization(Xtr)          # calibrate on train data only
    json_path = os.path.join(tmpdir, f'g{act_input_bits}.json')
    qm.export_graph_json(Xtr, json_path)

    captured = {}

    def _hook(i):
        def fn(_m, _in, out):
            captured[i] = out
        return fn

    handles = [b.core.act.register_forward_hook(_hook(i))
               for i, b in enumerate(qm.bundles)]
    with torch.no_grad():
        raw = qm(Xte)
        # With no softmax the last activation still returns a QuantTensor
        # (return_quant_tensor=True throughout, so the next bundle can consume the
        # scale directly) - unwrap it to compare in real units.
        brev_out = (raw.value if hasattr(raw, 'value') else raw).squeeze(-1)
    for h in handles:
        h.remove()

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    logits = model.forward(model.quantize_input(Xte.numpy()))

    # integer output -> real units
    last = model.bundles[model.bundle_order[-1]]
    pred = np.asarray(logits, dtype=np.float64).squeeze(-1) / 2 ** last['act_frac']
    rmse_int = float(np.sqrt(np.mean((pred - yte.numpy()) ** 2))) * ysd
    rmse_brev = float(torch.sqrt(nn.functional.mse_loss(brev_out, yte))) * ysd

    differing = total = worst = nbytes = 0
    for i, name in enumerate(model.bundle_order):
        lut = model.bundles[name]['lut']
        if lut is None:
            continue
        qt = captured[i]
        ref = np.rint(qt.value.detach().numpy() / qt.scale.item()).astype(np.int64)
        diff = np.abs(model.trace[name]['out'] - ref)
        differing += int((diff != 0).sum())
        total += diff.size
        worst = max(worst, int(diff.max()))
        nbytes += lut.nbytes

    return dict(rmse_int=rmse_int, rmse_brev=rmse_brev,
                mismatch=differing / total * 100 if total else 0.0,
                worst=worst, nbytes=nbytes,
                grids=[(b['lut'].in_bits, b['lut'].in_frac)
                       for b in model.bundles.values() if b['lut'] is not None])


def main(hidden=(64, 64, 64, 64), activations=('SiLU', 'Tanh')):
    import tempfile
    tmpdir = tempfile.mkdtemp()

    act_types = {'SiLU': nn.SiLU, 'Tanh': nn.Tanh, 'GELU': nn.GELU,
                 'Sigmoid': nn.Sigmoid, 'SELU': nn.SELU}
    arch = ' -> '.join(['5'] + [str(w) for w in hidden] + ['1'])

    for act_name in activations:
        act = act_types[act_name]
        net, Xtr, Xte, yte, ysd, rmse_f = train(act, hidden=hidden)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"\n{'=' * 78}")
        print(f"{act_name}  -  UCI Airfoil Self-Noise, {arch}  "
              f"({len(hidden)} hidden layers, {n_params} params)")
        print(f"{'=' * 78}")
        print(f"  float32 test RMSE: {rmse_f:.4f} dB   (target std 6.90 dB)")
        print(f"\n  {'config':<18} {'model RMSE':>11} {'hw RMSE':>9} {'hw-model':>9} "
              f"{'vs brevitas':>12} {'bytes':>7}")
        print("  " + "-" * 74)

        def row(label, r):
            gap = r['rmse_int'] - r['rmse_brev']
            flag = '' if r['mismatch'] == 0 else '  <- hw != model'
            print(f"  {label:<18} {r['rmse_brev']:>11.4f} {r['rmse_int']:>9.4f} "
                  f"{gap:>+9.4f} {r['mismatch']:>11.2f}% {r['nbytes']:>7}{flag}")

        row('1a (no in-quant)', evaluate(net, Xtr, Xte, yte, ysd, None, tmpdir))
        for bits in (4, 5, 6, 7, 8, 10, 12):
            row(f'1b @ {bits} bits', evaluate(net, Xtr, Xte, yte, ysd, bits, tmpdir))

    print("\n  model RMSE  : what the quantized model (brevitas) predicts")
    print("  hw RMSE     : what the integer pipeline actually computes")
    print("  hw-model    : the gap between them. Must be 0 for the golden "
          "reference to mean anything.")
    print("  vs brevitas : per-activation-value disagreement behind that gap")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 64, 64, 64],
                    help='hidden layer widths, e.g. --hidden 128 128 64 64')
    ap.add_argument('--act', nargs='+', default=['SiLU', 'Tanh'],
                    choices=['SiLU', 'Tanh', 'GELU', 'Sigmoid', 'SELU'])
    a = ap.parse_args()
    main(hidden=tuple(a.hidden), activations=a.act)
