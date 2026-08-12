"""End-to-end checks for variant 1b: quantizing a curved activation's input so a
small value LUT reproduces brevitas bit-exactly.

Everything here needs torch/brevitas, unlike test_brevitas_lut.py which is pure
numpy. The property under test cannot be checked without brevitas, because the
whole point of 1b is that it changes what brevitas itself computes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepsocflow.py.brevitas.ptq import quantized_model
from deepsocflow.py.brevitas.sim import FixedPointModel


X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])


def _curved():
    """Every activation that needs a table. Kept as a function so torch is only
    touched after the importorskip above."""
    import torch.nn as nn
    return {'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU,
            'sigmoid': nn.Sigmoid, 'selu': nn.SELU}


CURVED_NAMES = ['silu', 'tanh', 'gelu', 'sigmoid', 'selu']


def _net(activation=None):
    import torch.nn as nn

    act = activation or nn.SiLU
    torch.manual_seed(30)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_1 = nn.Linear(2, 8, bias=True)
            self.act_1 = act()
            self.hidden_2 = nn.Linear(8, 4, bias=True)
            self.act_2 = act()
            self.out = nn.Linear(4, 2, bias=True)

        def forward(self, x):
            x = self.act_1(self.hidden_1(x))
            x = self.act_2(self.hidden_2(x))
            return self.out(x)

    return Net()


def _run(act_input_bits, tmp_path, activation=None):
    """Quantize, export, and run both brevitas and the integer LUT pipeline over
    the same input. Returns (per-bundle brevitas activation ints, FixedPointModel)."""
    qm = quantized_model(_net(activation), weight_bits=8, bias_bits=16,
                         act_input_bits=act_input_bits)
    qm.quantization(X)
    json_path = str(tmp_path / f"graph_{act_input_bits}.json")
    qm.export_graph_json(X, json_path)

    captured = {}

    def _hook(idx):
        def fn(_m, _i, out):
            captured[idx] = out
        return fn

    handles = [b.core.act.register_forward_hook(_hook(i)) for i, b in enumerate(qm.bundles)]
    with torch.no_grad():
        qm(X)
    for h in handles:
        h.remove()

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    model.forward(model.quantize_input(X.numpy()))

    refs = {}
    for idx, name in enumerate(model.bundle_order):
        qt = captured[idx]
        refs[name] = np.rint(qt.value.detach().numpy() / qt.scale.item()).astype(np.int64)
    return refs, model


def _lut_disagreement(refs, model):
    """Per-value disagreement between the integer LUT pipeline and brevitas,
    counted only over bundles that actually use a table."""
    differing = total = worst = 0
    for name in model.bundle_order:
        if model.bundles[name]['lut'] is None:
            continue
        diff = np.abs(model.trace[name]['out'] - refs[name])
        differing += int((diff != 0).sum())
        total += diff.size
        worst = max(worst, int(diff.max()))
    return differing, total, worst


@pytest.mark.parametrize("name", CURVED_NAMES)
def test_1b_is_bit_exact_against_brevitas(name, tmp_path):
    """The claim that justifies variant 1b: with the activation's input
    quantized, a table on that same grid reproduces brevitas exactly.

    Run over every curved activation, not just silu - the shapes differ enough
    to break a fix that only happens to suit one of them. sigmoid is the
    sharpest case (unsigned output narrowed to bits-1 by ptq.py, fed by a signed
    accumulator), tanh/sigmoid saturate while silu/gelu/selu do not."""
    refs, model = _run(act_input_bits=8, tmp_path=tmp_path,
                       activation=_curved()[name])
    differing, total, worst = _lut_disagreement(refs, model)
    assert total > 0, "no LUT bundles were exercised - the test would be vacuous"
    assert (differing, worst) == (0, 0), (
        f"{name}: {differing}/{total} activation values disagree with brevitas "
        f"(worst {worst} LSB) - 1b should be bit-exact")


@pytest.mark.parametrize("name", CURVED_NAMES)
def test_1a_is_not_bit_exact_against_brevitas(name, tmp_path):
    """The other half of the claim, asserted rather than assumed: without
    input_quant the same table size does NOT reproduce brevitas, because
    brevitas is still evaluating the activation on the full-precision
    accumulator. If this ever starts passing, 1b has stopped buying anything and
    the extra quantization point should be removed."""
    refs, model = _run(act_input_bits=None, tmp_path=tmp_path,
                       activation=_curved()[name])
    differing, total, worst = _lut_disagreement(refs, model)
    assert total > 0
    assert differing > 0 and worst > 0, (
        f"{name}: 1a unexpectedly matched brevitas - if this is real, 1b's extra "
        f"quantization point is no longer paying for itself")


@pytest.mark.parametrize("name", CURVED_NAMES)
@pytest.mark.parametrize("bits", [4, 6, 8, 10])
def test_1b_is_bit_exact_at_every_input_width(name, bits, tmp_path):
    """Bit-exactness comes from the table being on the same grid brevitas
    quantized to - not from that grid being 8 bits wide. Narrower input_quant
    just means a smaller table (64 B at 6 bits), with the accuracy cost moving
    into the model where calibration can see it."""
    refs, model = _run(act_input_bits=bits, tmp_path=tmp_path,
                       activation=_curved()[name])
    differing, _, worst = _lut_disagreement(refs, model)
    assert (differing, worst) == (0, 0)
    for bundle in model.bundles.values():
        if bundle['lut'] is not None:
            assert bundle['lut'].in_bits == bits
            assert bundle['lut'].nbytes == 2 ** bits


def test_1b_table_stays_small(tmp_path):
    """1b's value is bit-exactness at a table size that fits in config_fw.h.
    8-bit input quant means 256 entries per activation, not the 64-128 KB that
    1a needs for the same guarantee."""
    _, model = _run(act_input_bits=8, tmp_path=tmp_path)
    luts = [b['lut'] for b in model.bundles.values() if b['lut'] is not None]
    assert luts
    for lut in luts:
        assert lut.in_bits == 8
        assert lut.nbytes == 256


def test_graph_json_carries_the_activation_input_grid(tmp_path):
    """sim.py sizes the table from act_in_bits/act_in_frac, so the exporter has
    to emit them - and must not emit them for 1a models, where no such grid
    exists."""
    import json

    qm = quantized_model(_net(), weight_bits=8, bias_bits=16, act_input_bits=8)
    qm.quantization(X)
    path_1b = str(tmp_path / "1b.json")
    qm.export_graph_json(X, path_1b)
    layers_1b = json.load(open(path_1b))['layers']
    curved = [c for c in layers_1b.values() if c['activation'] == 'silu']
    assert curved, "expected silu bundles in the exported graph"
    for cfg in curved:
        assert cfg['act_in_bits'] == 8
        assert isinstance(cfg['act_in_frac'], int)
        assert cfg['act_in_signed'] is True

    qm_1a = quantized_model(_net(), weight_bits=8, bias_bits=16)
    qm_1a.quantization(X)
    path_1a = str(tmp_path / "1a.json")
    qm_1a.export_graph_json(X, path_1a)
    for cfg in json.load(open(path_1a))['layers'].values():
        assert 'act_in_bits' not in cfg


def test_relu_gets_no_input_quant_even_when_1b_is_on(tmp_path):
    """input_quant is only for activations that need a table. Adding one to relu
    would create a second quantization point per bundle for no benefit - relu
    still runs on quant_lrelu."""
    import json
    import torch.nn as nn

    qm = quantized_model(_net(nn.ReLU), weight_bits=8, bias_bits=16, act_input_bits=8)
    qm.quantization(X)
    path = str(tmp_path / "relu.json")
    qm.export_graph_json(X, path)
    for cfg in json.load(open(path))['layers'].values():
        assert 'act_in_bits' not in cfg


def test_sigmoid_table_is_unsigned_but_indexed_signed(tmp_path):
    """sigmoid is the one activation where the two sides of the table disagree on
    signedness: ptq.py gives it an unsigned output narrowed to bits-1, while the
    accumulator addressing it is signed. Conflating them would send every
    negative input into the wrong half of the table, so it is pinned here rather
    than left to the general bit-exactness check to catch indirectly."""
    import torch.nn as nn

    _, model = _run(act_input_bits=8, tmp_path=tmp_path, activation=nn.Sigmoid)
    luts = [b['lut'] for b in model.bundles.values() if b['lut'] is not None]
    assert luts
    for lut in luts:
        assert lut.in_signed is True
        assert lut.out_signed is False
        assert lut.table.min() >= 0


# ---- hardware validation ----

def test_check_hardware_accepts_a_valid_lut_model(tmp_path):
    from deepsocflow.py.brevitas.export import check_hardware
    from deepsocflow.py.brevitas.hardware import Hardware

    _, model = _run(act_input_bits=8, tmp_path=tmp_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, axi_width=128)
    check_hardware(model, hw)  # must not raise


def test_check_hardware_rejects_a_lut_wider_than_x_bits(tmp_path):
    """A table whose entries do not fit the hardware's activation word would be
    truncated when packed into the .bin blob rather than raising - the same
    silent-corruption shape as the K_BITS/B_BITS gap closed on 2026-08-11."""
    from deepsocflow.py.brevitas.export import check_hardware
    from deepsocflow.py.brevitas.hardware import Hardware

    _, model = _run(act_input_bits=8, tmp_path=tmp_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, axi_width=128)
    # force one bundle's table to declare a wider output than the hardware holds
    lut = next(b['lut'] for b in model.bundles.values() if b['lut'] is not None)
    lut.out_bits = 16
    with pytest.raises(AssertionError, match="LUT out_bits"):
        check_hardware(model, hw)
