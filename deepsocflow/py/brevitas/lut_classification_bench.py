"""Benchmark: does a LUT activation's error change what the deployed model answers?

The companion to lut_regression_bench.py, asking the question a regression target
cannot. On a continuous output, activation error shows up as RMSE - proportional,
visible, easy to reason about. On a classifier it is hidden until it is not: an
argmax absorbs small errors completely, right up to the point where one flips a
label. So the number that matters here is not accuracy, it is **flips** - the
fraction of inputs where the hardware picks a different class than the quantized
model said it would.

XOR could not measure this either: 4 rows, probabilities pinned at 0/1. This uses
UCI Letter Recognition (20000 x 16 -> 26 classes), where decisions are genuinely
close and there are 25 wrong classes for a flip to land on.

Run:  python -m deepsocflow.py.brevitas.lut_classification_bench
      python -m deepsocflow.py.brevitas.lut_classification_bench --hidden 128 128 64
"""

import contextlib
import io
import os
import tempfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from deepsocflow.py.brevitas.ptq import quantized_model
from deepsocflow.py.brevitas.sim import FixedPointModel

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "letter-recognition/letter-recognition.data")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "letter-recognition.data")
SEED = 0
N_CLASSES = 26


def fetch():
    if not os.path.exists(DATA):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, DATA)
    return DATA


def load():
    rows = [l.strip().split(',') for l in open(fetch()) if l.strip()]
    y = np.array([ord(r[0]) - ord('A') for r in rows], dtype=np.int64)
    X = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = perm[:n_tr], perm[n_tr:]

    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xs = ((X - mu) / sd).astype(np.float32)
    return (torch.tensor(Xs[tr]), torch.tensor(y[tr]),
            torch.tensor(Xs[te]), torch.tensor(y[te]))


class MLP(nn.Module):
    """Attributes registered in execution order, ending in Softmax - the shape
    ptq.py's quantized_model expects (compute -> activation -> ... -> Softmax)."""

    def __init__(self, act=nn.SiLU, hidden=(64, 64, 64), in_features=16):
        super().__init__()
        prev = in_features
        for i, w in enumerate(hidden, start=1):
            setattr(self, f'hidden_{i}', nn.Linear(prev, w, bias=True))
            setattr(self, f'act_{i}', act())
            prev = w
        self.out = nn.Linear(prev, N_CLASSES, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.n_hidden = len(hidden)

    def forward(self, x):
        for i in range(1, self.n_hidden + 1):
            x = getattr(self, f'act_{i}')(getattr(self, f'hidden_{i}')(x))
        return self.softmax(self.out(x))


def train(act=nn.SiLU, hidden=(64, 64, 64), epochs=60, batch=256, lr=2e-3):
    Xtr, ytr, Xte, yte = load()
    torch.manual_seed(SEED)
    net = MLP(act, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.nll_loss(
                torch.log(net(Xtr[idx]) + 1e-12), ytr[idx]).backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        acc = float((net(Xte).argmax(-1) == yte).float().mean())
    return net, Xtr, Xte, yte, acc


def evaluate(net, Xtr, Xte, yte, act_input_bits, tmpdir, calib=512):
    with contextlib.redirect_stdout(io.StringIO()):
        qm = quantized_model(net, weight_bits=8, bias_bits=16,
                             act_input_bits=act_input_bits)
        qm.quantization(Xtr[:calib])
    json_path = os.path.join(tmpdir, f'g{act_input_bits}.json')
    with contextlib.redirect_stdout(io.StringIO()):
        qm.export_graph_json(Xtr[:1], json_path)

    captured = {}

    def _hook(i):
        def fn(_m, _in, out):
            captured[i] = out
        return fn

    handles = [b.core.act.register_forward_hook(_hook(i))
               for i, b in enumerate(qm.bundles)]
    with torch.no_grad():
        raw = qm(Xte)
        probs = (raw.value if hasattr(raw, 'value') else raw)
    for h in handles:
        h.remove()
    model_pred = probs.argmax(-1).numpy()

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    logits = model.forward(model.quantize_input(Xte.numpy()))
    hw_pred = np.asarray(logits).argmax(-1)

    y = yte.numpy()
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

    return dict(
        acc_model=float((model_pred == y).mean()),
        acc_hw=float((hw_pred == y).mean()),
        # The number this benchmark exists for: inputs where the hardware answers
        # a different class than the quantized model predicted it would.
        flips=float((hw_pred != model_pred).mean()),
        mismatch=differing / total * 100 if total else 0.0,
        worst=worst, nbytes=nbytes)


def main(hidden=(64, 64, 64), activations=('SiLU', 'Tanh'),
         widths=(4, 6, 8, 10, 12)):
    tmpdir = tempfile.mkdtemp()
    act_types = {'SiLU': nn.SiLU, 'Tanh': nn.Tanh, 'GELU': nn.GELU,
                 'Sigmoid': nn.Sigmoid, 'SELU': nn.SELU}
    arch = ' -> '.join(['16'] + [str(w) for w in hidden] + [str(N_CLASSES)])

    for act_name in activations:
        net, Xtr, Xte, yte, acc_f = train(act_types[act_name], hidden=hidden)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"\n{'=' * 80}")
        print(f"{act_name}  -  UCI Letter Recognition, {arch}  "
              f"({len(hidden)} hidden layers, {n_params} params)")
        print(f"{'=' * 80}")
        print(f"  float32 test accuracy: {acc_f * 100:.2f}%  "
              f"({len(Xte)} held-out samples, 26 classes)")
        print(f"\n  {'config':<18} {'model acc':>10} {'hw acc':>8} {'FLIPS':>8} "
              f"{'vs brevitas':>12} {'bytes':>7}")
        print("  " + "-" * 70)

        def row(label, r):
            flag = '' if r['flips'] == 0 else '  <- hw answers differently'
            print(f"  {label:<18} {r['acc_model'] * 100:>9.2f}% "
                  f"{r['acc_hw'] * 100:>7.2f}% {r['flips'] * 100:>7.2f}% "
                  f"{r['mismatch']:>11.2f}% {r['nbytes']:>7}{flag}")

        row('1a (no in-quant)', evaluate(net, Xtr, Xte, yte, None, tmpdir))
        for bits in widths:
            row(f'1b @ {bits} bits',
                evaluate(net, Xtr, Xte, yte, bits, tmpdir))

    print("\n  model acc : accuracy of the quantized model (brevitas)")
    print("  hw acc    : accuracy of the integer pipeline actually deployed")
    print("  FLIPS     : inputs where hw picks a DIFFERENT class than the model "
          "said it would.")
    print("              Not the same as an accuracy gap - flips can cancel out "
          "and leave")
    print("              accuracy looking fine while individual answers differ.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 64, 64])
    ap.add_argument('--act', nargs='+', default=['SiLU', 'Tanh'],
                    choices=['SiLU', 'Tanh', 'GELU', 'Sigmoid', 'SELU'])
    a = ap.parse_args()
    main(hidden=tuple(a.hidden), activations=a.act)
