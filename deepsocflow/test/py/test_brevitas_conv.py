"""Conv2d support on the brevitas backend (sim.py / ptq.py / adapter.py).

The load-bearing test here is test_matches_exporters_float_conv: sim.py's integer
convolution and rtl_export.py's float one are two independent implementations of
the same operation, and every bit-exactness claim in this backend rests on them
agreeing. The rest guard the layout permutation and the geometry asserts, which
are the parts that fail silently rather than loudly.
"""
import types

import numpy as np
import pytest

from deepsocflow.py.brevitas.sim import (
    conv2d_same_int, to_hwio, _apply_conv_stride,
)
from deepsocflow.py.brevitas.rtl_export import _conv2d_same


def _rand_int(rng, shape, lo=-8, hi=8):
    return rng.integers(lo, hi, size=shape, dtype=np.int64)


@pytest.mark.parametrize("kh,kw", [(1, 1), (3, 3), (5, 5), (1, 3), (3, 1)])
def test_matches_exporters_float_conv(kh, kw):
    # rtl_export.py::_conv2d_same computes the golden per-pass sums the RTL is
    # checked against; sim.py::conv2d_same_int computes what the model predicts.
    # If these two ever disagree, every "Error: 0" in this backend is meaningless.
    # Values are kept small so the exporter's float32 accumulator is exact and
    # the comparison tests the convolution, not the mantissa.
    rng = np.random.default_rng(0)
    x = _rand_int(rng, (2, 8, 7, 3))
    w = _rand_int(rng, (kh, kw, 3, 4))

    ours = conv2d_same_int(x, w)
    theirs = _conv2d_same(x.astype(np.float32), w.astype(np.float32))

    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs.astype(np.int64))


def test_stays_integer():
    # The exporter may accumulate in float32; the model must not - it is the
    # reference the hardware is judged against, so it cannot inherit a mantissa
    # limit. Values here exceed float32's exact-integer range (2**24).
    x = np.full((1, 8, 8, 4), 1 << 13, dtype=np.int64)
    w = np.full((3, 3, 4, 1), 1 << 12, dtype=np.int64)
    out = conv2d_same_int(x, w)
    assert out.dtype == np.int64
    # Centre pixel sees the full 3x3x4 window, none of it clipped by padding.
    assert out[0, 4, 4, 0] == 9 * 4 * (1 << 13) * (1 << 12)


def test_padding_alignment_is_centred():
    # For an odd kernel 'same' padding is symmetric (pad_total = K-1 = 2, split
    # 1/1), so there is no bottom/right asymmetry to pin here - what matters is
    # that the window is centred rather than shifted by one, which is the
    # failure a wrong split produces: same output shape, plausible numbers, every
    # feature map off by a pixel.
    x = np.zeros((1, 3, 3, 1), dtype=np.int64)
    x[0, 0, 0, 0] = 1  # top-left corner only
    w = np.arange(1, 10, dtype=np.int64).reshape(3, 3, 1, 1)

    out = conv2d_same_int(x, w)
    assert out.shape == (1, 3, 3, 1)
    # Centred: output (0,0) sits over input (0,0), so the single non-zero pixel
    # meets the kernel's centre tap, w[1,1] == 5. Had the padding gone 0/2
    # instead of 1/1 this would be w[0,0] == 1.
    assert out[0, 0, 0, 0] == 5


def test_to_hwio_permutes_torch_weight_layout():
    # torch stores (CO, CI, KH, KW); the engine and dataflow.py's reorder
    # helpers read (KH, KW, CI, CO).
    w = np.arange(2 * 3 * 5 * 7, dtype=np.int64).reshape(2, 3, 5, 7)
    out = to_hwio(w)
    assert out.shape == (5, 7, 3, 2)
    for co in range(2):
        for ci in range(3):
            for kh in range(5):
                for kw in range(7):
                    assert out[kh, kw, ci, co] == w[co, ci, kh, kw]


def test_stride_1_is_identity():
    rng = np.random.default_rng(1)
    y = _rand_int(rng, (1, 8, 8, 2))
    assert np.array_equal(_apply_conv_stride(y, (1, 1), (3, 3)), y)


def test_stride_start_offset_matches_dataflow_formula():
    # dataflow.py:44-46 derives CSH_SHIFT/CSW_SHIFT; for a 3x3 stride-2 conv over
    # an 8-wide axis that offset is 1, not 0. torch's own Conv2d(stride=2,
    # padding=1) anchors at 0 instead, so a float model built with plain
    # symmetric padding does NOT match this hardware - hence the explicit check.
    y = np.arange(8 * 8, dtype=np.int64).reshape(1, 8, 8, 1)
    out = _apply_conv_stride(y, (2, 2), (3, 3))
    assert out.shape == (1, 4, 4, 1)
    assert np.array_equal(out[0, :, 0, 0], y[0, 1::2, 1, 0])


def test_conv_geometry_accepts_same_padding_both_spellings():
    from deepsocflow.py.brevitas.ptq import _conv_geometry

    as_string = _conv_geometry(types.SimpleNamespace(
        kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), groups=1,
        padding='same', in_channels=4, out_channels=8))
    # torch rejects padding='same' on a strided conv, so a strided layer has to
    # spell the same thing as an int; both must normalize identically.
    as_int = _conv_geometry(types.SimpleNamespace(
        kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), groups=1,
        padding=(1, 1), in_channels=4, out_channels=8))

    assert as_string['padding'] == as_int['padding'] == [1, 1]
    assert as_string['stride'] == [1, 1] and as_int['stride'] == [2, 2]
    assert as_string['in_channels'] == 4 and as_string['out_channels'] == 8


@pytest.mark.parametrize("bad,match", [
    (dict(padding=(0, 0)), "not 'same'"),
    (dict(padding='valid'), "not supported"),
    (dict(kernel_size=(2, 2), padding=(1, 1)), "odd"),
    (dict(dilation=(2, 2)), "dilation"),
    (dict(groups=2), "groups"),
])
def test_conv_geometry_rejects_what_the_engine_cannot_run(bad, match):
    # Each of these would otherwise run and produce silently wrong output rather
    # than fail - the engine implements exactly one convolution shape.
    from deepsocflow.py.brevitas.ptq import _conv_geometry

    core = dict(kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), groups=1,
                padding='same', in_channels=4, out_channels=8)
    core.update(bad)
    with pytest.raises(AssertionError, match=match):
        _conv_geometry(types.SimpleNamespace(**core))


def test_check_hardware_counts_kernel_area_in_acc_width():
    # A conv accumulates over KH*KW*CI taps. Counting only CI under-reports the
    # accumulator width by clog2(KH*KW) - 4 bits for a 3x3 - and would let a
    # model through that overflows the engine's accumulator.
    from deepsocflow.py.brevitas.export import check_hardware

    hw = types.SimpleNamespace(X_BITS=8, K_BITS=8, B_BITS=16, Y_BITS=20)
    model = types.SimpleNamespace(
        bundle_order=['b0'],
        bundles={'b0': dict(
            type='conv', conv=dict(kernel_size=(3, 3)),
            input_bits=8, act_bits=8, weight_bits=8, bias_bits=16,
            in_features=64, act_signed=True, lut=None,
            input_frac=7, weight_frac=6)},
        trace={})

    # 8 + 8 + clog2(9*64=576) = 26 > Y_BITS=20, and > 24 too.
    with pytest.raises(AssertionError, match="ACC_WIDTH"):
        check_hardware(model, hw)


def test_pad_single_input_channel_is_exact_and_targeted():
    # CI == 1 makes the engine's C_CI counter degenerate (max_in == 0), which
    # deadlocks or miscomputes depending on the bundle - see CLAUDE.md. Padding
    # to two channels with zeros fixes it without changing any output value,
    # which is the property worth pinning: a padding that altered the numbers
    # would trade a loud failure for a quiet one.
    from deepsocflow.py.brevitas.adapter import pad_single_input_channel
    from deepsocflow.py.brevitas.sim import conv2d_same_int

    rng = np.random.default_rng(3)
    w = _rand_int(rng, (3, 3, 1, 4))          # (KH,KW,CI,CO), CI == 1
    x = _rand_int(rng, (2, 6, 6, 1))          # (N,H,W,CI)

    w_p, x_p = pad_single_input_channel(w, x, is_conv=True)
    assert w_p.shape == (3, 3, 2, 4) and x_p.shape == (2, 6, 6, 2)
    assert np.array_equal(conv2d_same_int(x, w), conv2d_same_int(x_p, w_p))

    # Dense layers index the channel axis differently, and a one-feature dense
    # layer is the same degenerate case.
    w_d = _rand_int(rng, (1, 5))               # (CI,CO)
    x_d = _rand_int(rng, (4, 1))               # (N,CI)
    w_dp, x_dp = pad_single_input_channel(w_d, x_d, is_conv=False)
    assert w_dp.shape == (2, 5) and x_dp.shape == (4, 2)
    assert np.array_equal(x_d @ w_d, x_dp @ w_dp)


def test_pad_single_input_channel_leaves_normal_layers_alone():
    from deepsocflow.py.brevitas.adapter import pad_single_input_channel

    rng = np.random.default_rng(4)
    w = _rand_int(rng, (3, 3, 8, 4))
    x = _rand_int(rng, (2, 6, 6, 8))
    w_p, x_p = pad_single_input_channel(w, x, is_conv=True)
    assert w_p is w and x_p is x


def _c_div_round_table():
    """Compiles and runs deepsocflow/test/c/div_round_dump.c, returning its rows.

    Pinning against the compiled macro rather than against a Python reading of it
    is the point: the tie-break is idiosyncratic enough that a transcription bug
    would look like a plausible rounding choice.
    """
    import pathlib, shutil, subprocess, tempfile

    src = pathlib.Path(__file__).resolve().parents[1] / 'c' / 'div_round_dump.c'
    cc = shutil.which('cc') or shutil.which('gcc')
    if cc is None or not src.exists():
        pytest.skip("no C compiler or div_round_dump.c not present")

    with tempfile.TemporaryDirectory() as tmp:
        exe = pathlib.Path(tmp) / 'div_round_dump'
        subprocess.run([cc, '-O2', '-o', str(exe), str(src)], check=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return [tuple(int(v) for v in line.split()) for line in out.splitlines()]


def test_div_round_matches_c():
    from deepsocflow.py.brevitas.sim import div_round

    rows = _c_div_round_table()
    assert len(rows) > 1000, "the dump looks truncated"

    a = np.array([r[0] for r in rows], dtype=np.int64)
    b_vals = sorted({r[1] for r in rows})
    expected = np.array([r[2] for r in rows], dtype=np.int64)

    got = np.empty_like(expected)
    for b in b_vals:
        m = np.array([r[1] == b for r in rows])
        got[m] = div_round(a[m], b)

    bad = np.flatnonzero(got != expected)
    assert bad.size == 0, (
        f"{bad.size} mismatches, first at a={a[bad[0]]} b={rows[bad[0]][1]}: "
        f"C says {expected[bad[0]]}, python says {got[bad[0]]}")


def test_div_round_result_always_fits_the_activation_width():
    # This is what makes runtime.h's avg-pool activation bug (the pa_* result is
    # computed on the wrong variable and discarded) harmless: the clip it drops
    # would never have fired. An average of in-range values is in range - but the
    # tie-break is odd enough that it is measured here rather than argued.
    from deepsocflow.py.brevitas.sim import div_round

    lo, hi = -128, 127
    for count in (1, 2, 3, 4, 6, 9, 12, 16, 25, 36):
        sums = np.arange(lo * count, hi * count + 1, dtype=np.int64)
        out = div_round(sums, count)
        assert out.min() >= lo and out.max() <= hi, (
            f"count={count}: div_round produced [{out.min()},{out.max()}], "
            f"outside [{lo},{hi}]")


def test_avgpool_uses_div_round_not_a_mean():
    # The engine sums the window and applies div_round; an ordinary mean disagrees
    # with it by up to 1 LSB, almost always on negative values. Pinning this stops
    # a "cleaner" reimplementation from silently breaking bit-exactness.
    from deepsocflow.py.brevitas.sim import avgpool2d_valid_int, div_round

    rng = np.random.default_rng(7)
    x = _rand_int(rng, (2, 4, 4, 3), lo=-128, hi=128)
    out = avgpool2d_valid_int(x, (2, 2), (2, 2))
    assert out.shape == (2, 2, 2, 3)

    for n in range(2):
        for i in range(2):
            for j in range(2):
                for c in range(3):
                    window = x[n, i * 2:i * 2 + 2, j * 2:j * 2 + 2, c]
                    assert out[n, i, j, c] == div_round(window.sum(), 4)

    # And it is genuinely not the mean: a window summing to -4 over 4 elements
    # gives 0 under div_round, where a mean gives -1.
    assert div_round(np.int64(-4), 4) == 0


def test_avgpool_torch_and_numpy_paths_agree():
    # sim.py drives the hardware comparison; the torch module drives the float
    # model. If they disagree, sim-vs-brevitas can never be exactly zero, which is
    # the property every conv stage asserts.
    import torch
    from deepsocflow.py.brevitas.sim import div_round
    from deepsocflow.py.brevitas.xlayer.quantPooling import div_round_torch

    rng = np.random.default_rng(11)
    for count in (1, 4, 9, 16):
        sums = rng.integers(-128 * count, 127 * count, size=500, dtype=np.int64)
        ours = div_round(sums, count)
        theirs = div_round_torch(torch.tensor(sums, dtype=torch.float64), count)
        assert np.array_equal(ours, theirs.numpy().astype(np.int64))


def test_batchnorm_fold_is_numerically_equivalent():
    # The one property everything else rests on: the folded layer must compute
    # what conv-then-BN computed. If this drifts, every downstream "Error: 0"
    # is measuring agreement with the wrong model.
    import torch
    import torch.nn as nn
    from deepsocflow.py.brevitas.ptq import _fold_batchnorm

    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding='same', bias=False)
    bn = nn.BatchNorm2d(8)
    # Give the BN non-trivial statistics; a freshly built one is the identity.
    bn.running_mean.normal_(0, 1)
    bn.running_var.uniform_(0.5, 2.0)
    bn.weight.data.normal_(1, 0.3)
    bn.bias.data.normal_(0, 0.3)
    conv.eval(), bn.eval()

    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        reference = bn(conv(x))
        folded = _fold_batchnorm(conv, bn)(x)

    assert torch.allclose(reference, folded, atol=1e-5), (
        f"max abs err {float((reference - folded).abs().max())}")


def test_batchnorm_fold_creates_a_bias_on_a_bias_free_conv():
    # A conv followed by BN is nearly always built with bias=False; the fold is
    # what produces the bias, and _quantize_layer's has_bias path then has to
    # pick it up. That handover is the part that would silently drop the BN's
    # shift term.
    import torch
    import torch.nn as nn
    from deepsocflow.py.brevitas.ptq import _fold_batchnorm, _quantize_layer

    torch.manual_seed(1)
    conv = nn.Conv2d(3, 8, 3, padding='same', bias=False).eval()
    bn = nn.BatchNorm2d(8).eval()
    bn.bias.data.normal_(0, 1)

    assert conv.bias is None
    fused = _fold_batchnorm(conv, bn)
    assert fused.bias is not None
    assert torch.allclose(fused.bias, bn.bias, atol=1e-5), (
        "with zero running_mean and unit running_var the folded bias is just beta")

    core = _quantize_layer(fused, weight_bits=8, bias_bits=16)
    assert core.bias is not None, "the folded bias must survive quantization"
    assert core.configured_bias_bits == 16


def test_batchnorm_fold_requires_eval_mode():
    import torch.nn as nn
    from deepsocflow.py.brevitas.ptq import quantized_model

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_1 = nn.Conv2d(3, 4, 3, padding='same', bias=False)
            self.bn_1 = nn.BatchNorm2d(4)
            self.relu_1 = nn.ReLU()

        def forward(self, x):
            return self.relu_1(self.bn_1(self.conv_1(x)))

    with pytest.raises(AssertionError, match="train mode"):
        quantized_model(Net(), weight_bits=8, bias_bits=16)   # left in train mode


def test_stage_h_actually_exercises_the_fold():
    # A freshly built BatchNorm has running_mean=0, running_var=1, gamma=1,
    # beta=0, which makes folding the identity - so a BN stage built without
    # priming passes while testing nothing. This guards that specific vacuity,
    # which is easy to reintroduce and invisible from a green run.
    import torch
    from deepsocflow.py.brevitas.conv import build_model, stage_data, prime_batchnorm
    from deepsocflow.py.brevitas.ptq import _fold_batchnorm

    X, _, _ = stage_data('h')
    model = prime_batchnorm(build_model('h'), X)

    folded = _fold_batchnorm(model.conv_1, model.bn_1)
    assert not torch.allclose(folded.weight, model.conv_1.weight), (
        "folding left the weights untouched - the BatchNorm statistics are still "
        "at their defaults, so this stage is not testing the fold")
    assert folded.bias.abs().max() > 0


@pytest.mark.parametrize("n,k,s", [(8, 3, 2), (16, 3, 2), (224, 7, 2), (56, 3, 2), (7, 3, 2)])
def test_tf_same_padding_reproduces_the_engines_strided_conv(n, k, s):
    # The engine convolves at stride 1 with symmetric padding and then keeps
    # pixels from CSH_SHIFT; torch's Conv2d(padding=k//2, stride=s) keeps them
    # from 0 instead. Padding explicitly with TF's asymmetric split makes a
    # padding=0 strided conv land on the engine's pixels. Sizes include ResNet18's
    # real geometries (224/7/2 stem, 56/3/2 stage transition).
    import torch
    import torch.nn.functional as F
    from deepsocflow.py.brevitas.ptq import tf_same_padding
    from deepsocflow.py.brevitas.sim import conv2d_same_int, _apply_conv_stride, to_hwio

    rng = np.random.default_rng(0)
    x = _rand_int(rng, (1, n, n, 4))
    w = _rand_int(rng, (2, 4, k, k))          # torch layout (CO,CI,KH,KW)

    engine = _apply_conv_stride(conv2d_same_int(x, to_hwio(w)), (s, s), (k, k))

    lo, hi = tf_same_padding(n, k, s)
    xt = F.pad(torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float64), (lo, hi, lo, hi))
    torch_out = F.conv2d(xt, torch.tensor(w, dtype=torch.float64), stride=s)
    torch_out = torch_out.numpy().transpose(0, 2, 3, 1).astype(np.int64)

    assert engine.shape == torch_out.shape
    assert np.array_equal(engine, torch_out)


def test_explicit_pad_must_match_the_engines_split():
    # A pad that is merely plausible - symmetric, right total - still shifts every
    # feature map. The check has to be on the exact split, not on the total.
    import types
    from deepsocflow.py.brevitas.ptq import _assert_explicit_pad_matches_engine, tf_same_padding

    lo, hi = tf_same_padding(8, 3, 2)
    assert (lo, hi) == (0, 1), "TF's split for 8/3/2 puts nothing on top"

    good = dict(kernel_size=[3, 3], stride=[2, 2], explicit_pad=[lo, hi, lo, hi])
    _assert_explicit_pad_matches_engine(good, 8, 8)          # must not raise

    symmetric_but_wrong = dict(kernel_size=[3, 3], stride=[2, 2], explicit_pad=[1, 1, 1, 1])
    with pytest.raises(AssertionError, match="explicit pad"):
        _assert_explicit_pad_matches_engine(symmetric_but_wrong, 8, 8)


def test_residual_bundle_output_frac_is_the_add_activations():
    # A bundle with a skip ends on its ADD activation's grid, not its core
    # activation's, so that is the frac the next bundle must be told about.
    # Recording act_frac instead makes the consumer apply a spurious shift - and
    # the bug is invisible whenever the residual bundle happens to be last,
    # because nothing consumes its output. Hence the trailing conv here.
    import torch
    import torch.nn as nn
    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.sim import FixedPointModel

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_1 = nn.Conv2d(3, 8, 3, padding='same', bias=True)
            self.relu_1 = nn.ReLU()
            self.conv_2 = nn.Conv2d(8, 8, 3, padding='same', bias=True)
            self.relu_2 = nn.ReLU()
            self.conv_3 = nn.Conv2d(8, 2, 1, bias=True)      # consumes the residual bundle

        def forward(self, x):
            skip = self.relu_1(self.conv_1(x))
            return self.conv_3(self.relu_2(self.conv_2(skip)) + skip)

    torch.manual_seed(0)
    x = (torch.rand(32, 3, 8, 8) > 0.5).float()
    qm = quantized_model(Net().eval(), weight_bits=8, bias_bits=16,
                         residuals={'conv_2': 'conv_1'})
    qm.quantization(x)
    qm.eval()

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'g.json')
        qm.export_graph_json(x[:4], path)
        fp = FixedPointModel(path)
        fp.load_int_weights(path)
        out = fp.forward(fp.quantize_input(x[:4]))

    with torch.no_grad():
        ref = qm(x[:4])
    ref = (ref.value if hasattr(ref, 'value') else ref).detach().numpy()
    last = fp.bundles[fp.bundle_order[-1]]
    got = (out.astype('float64') / 2 ** last['act_frac']).transpose(0, 3, 1, 2)

    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())}/{got.size} values differ - the bundle after a "
        f"residual is reading its input on the wrong fractional grid")
