import math

import numpy as np
import pytest

from deepsocflow.py.brevitas.lut import (
    ActLut, CURVED_ACTIVATIONS, exact_activation, levels, lut_activation, mismatch,
)


# ---- raw two's-complement indexing ----

def test_levels_are_in_raw_twos_complement_order():
    # The whole point of the pre-permutation: masking a signed value with the
    # index mask must recover that value's own table slot, with no bias add.
    for bits in (4, 8):
        lv = levels(bits, signed=True)
        for v in range(-2 ** (bits - 1), 2 ** (bits - 1)):
            assert lv[v & (2 ** bits - 1)] == v
        # positive half first, negative half second
        assert lv[0] == 0
        assert lv[2 ** (bits - 1)] == -2 ** (bits - 1)


def test_unsigned_levels_are_identity():
    assert levels(4, signed=False).tolist() == list(range(16))


def test_lookup_indexes_by_raw_bits():
    lut = ActLut.for_bundle('identity', act_bits=8, act_frac=4, act_signed=True)
    # identity at matching grids means table[v] == v, addressed by raw bits
    for v in (-128, -1, 0, 1, 127):
        assert lut.lookup(np.int64(v)) == v


# ---- the table agrees with the float reference at its own grid points ----

def test_table_is_exact_at_its_own_grid_points():
    # When the accumulator already sits on the table's index grid (shift of 0),
    # the LUT is exact by construction for every representable input. This is
    # the property that makes a value LUT bit-exact rather than approximate.
    for activation in sorted(CURVED_ACTIVATIONS):
        lut = ActLut(activation, in_bits=8, in_frac=4, in_signed=True,
                     out_bits=8, out_frac=4,
                     out_signed=activation not in ('sigmoid',))
        acc = levels(8, signed=True)
        got = lut_activation(acc, acc_frac=4, lut=lut)
        ref = exact_activation(acc, 4, activation, lut.out_bits, lut.out_frac,
                               lut.out_signed)
        assert np.array_equal(got, ref), activation


def test_identity_lut_reproduces_shift_and_clip():
    # An identity LUT must agree with what the non-LUT path (shift_round + clip)
    # would produce, otherwise the two branches of forward() disagree.
    from deepsocflow.py.brevitas.sim import shift_round
    lut = ActLut('identity', in_bits=8, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=4, out_signed=True)
    acc = np.arange(-4000, 4000, dtype=np.int64)
    got = lut_activation(acc, acc_frac=10, lut=lut)
    ref = np.clip(shift_round(acc, 10 - 4), -128, 127)
    assert np.array_equal(got, ref)


def test_relu_lut_reproduces_shift_and_clip():
    from deepsocflow.py.brevitas.sim import shift_round
    lut = ActLut('relu', in_bits=8, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=4, out_signed=False)
    acc = np.arange(-4000, 4000, dtype=np.int64)
    got = lut_activation(acc, acc_frac=10, lut=lut)
    ref = np.clip(shift_round(np.clip(acc, 0, None), 10 - 4), 0, 255)
    assert np.array_equal(got, ref)


def test_table_saturates_instead_of_wrapping():
    # tanh at frac 7 in 8 signed bits: +1.0 is not representable (max is 127/128),
    # so the table must clip to 127 rather than wrap to -128.
    lut = ActLut('tanh', in_bits=8, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=7, out_signed=True)
    assert lut.table.max() == 127
    assert lut.table.min() >= -128


# ---- power-of-two guards ----

def test_index_shift_is_a_right_shift():
    lut = ActLut.for_bundle('silu', act_bits=8, act_frac=4, act_signed=True)
    assert lut.index_shift(14) == 10
    assert lut.index_shift(4) == 0


def test_index_shift_rejects_a_left_shift():
    # acc_frac < in_frac would mean the table is finer than the accumulator that
    # feeds it - the rescale stops being a pure right shift, which is exactly the
    # property that keeps this multiplier-free.
    lut = ActLut.for_bundle('silu', act_bits=8, act_frac=6, act_signed=True)
    with pytest.raises(ValueError, match="right shift"):
        lut.index_shift(4)


def test_non_integer_frac_rejected():
    with pytest.raises(ValueError, match="power of two"):
        ActLut('silu', in_bits=8, in_frac=4.5, in_signed=True,
               out_bits=8, out_frac=4, out_signed=True)


def test_leaky_relu_slope_must_be_power_of_two():
    # Mirrors ptq.py::_quantize_activation and adapter.py::act_params. A LUT
    # could represent any slope, but the rest of the pipeline still cannot
    # deploy one, so accepting it here would only move the failure later.
    with pytest.raises(ValueError, match="power of two"):
        ActLut('leaky_relu', in_bits=8, in_frac=4, in_signed=True,
               out_bits=8, out_frac=4, out_signed=True, negative_slope=0.1)

    lut = ActLut('leaky_relu', in_bits=8, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=4, out_signed=True, negative_slope=0.125)
    assert lut.lookup(np.int64(-8)) == round(-8 * 0.125)


def test_oversized_table_refused():
    with pytest.raises(ValueError, match="refusing"):
        ActLut('silu', in_bits=25, in_frac=4, in_signed=True,
               out_bits=8, out_frac=4, out_signed=True)


def test_unknown_activation_rejected():
    with pytest.raises(ValueError, match="no float reference"):
        ActLut('mish', in_bits=8, in_frac=4, in_signed=True,
               out_bits=8, out_frac=4, out_signed=True)


# ---- reported cost ----

def test_nbytes_matches_table_size():
    assert ActLut('silu', 8, 4, True, 8, 4, True).nbytes == 256
    assert ActLut('silu', 10, 6, True, 8, 4, True).nbytes == 1024


def test_input_range_reports_addressable_span():
    lut = ActLut('tanh', in_bits=8, in_frac=7, in_signed=True,
                 out_bits=8, out_frac=7, out_signed=True)
    lo, hi = lut.input_range
    # frac 7 in 8 signed bits addresses only [-1, 1) - far too narrow for tanh,
    # which is the concrete failure mode of applying variant 1a to a saturating
    # activation. Asserted here so the limitation is pinned, not just documented.
    assert (lo, hi) == (-1.0, pytest.approx(127 / 128))


# ---- 1a default vs override ----

def test_for_bundle_defaults_index_grid_to_output_grid():
    lut = ActLut.for_bundle('silu', act_bits=8, act_frac=4, act_signed=True)
    assert (lut.in_bits, lut.in_frac) == (8, 4)


def test_for_bundle_index_is_signed_even_when_output_is_not():
    # sigmoid's output is unsigned, but the accumulator feeding the table is
    # signed - conflating the two would make every negative input index into the
    # wrong half of the table.
    lut = ActLut.for_bundle('sigmoid', act_bits=7, act_frac=7, act_signed=False,
                            in_bits=8, in_frac=4)
    assert lut.in_signed is True
    assert lut.out_signed is False
    assert lut.lookup(np.int64(-64)) == pytest.approx(
        round(1 / (1 + math.exp(4.0)) * 128), abs=1)


# ---- mismatch measurement ----

def test_mismatch_is_zero_when_grids_match():
    lut = ActLut('silu', in_bits=8, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=4, out_signed=True)
    frac, worst = mismatch(levels(8, signed=True), 4, lut)
    assert frac == 0.0 and worst == 0


@pytest.mark.parametrize("activation,domain_bits", [
    ('silu', 3), ('gelu', 3), ('selu', 3), ('tanh', 2), ('sigmoid', 3),
])
def test_bit_exact_only_at_full_accumulator_resolution(activation, domain_bits):
    """The governing law of this approach, pinned as a test.

    A value LUT reproduces the exact quantized activation if and only if its
    index grid loses no accumulator detail (in_frac == acc_frac) and covers the
    accumulator's whole range (no saturation at the table edges). Anything
    coarser is an approximation, however large the table.

    This is the property that decides whether the LUT path can keep the
    pipeline's existing bit-exact ("Error: 0") standard, so it is asserted in
    both directions - exact at full resolution, not exact one bit below it."""
    acc_frac = 13
    out = dict(act_bits=8, act_frac=7 if activation in ('tanh', 'sigmoid') else 4,
               act_signed=activation != 'sigmoid')
    half = 2 ** domain_bits
    acc = np.linspace(-half * 2 ** acc_frac, half * 2 ** acc_frac, 20001).astype(np.int64)

    exact = ActLut.for_bundle(activation, in_bits=acc_frac + domain_bits + 1,
                              in_frac=acc_frac, **out)
    frac, worst = mismatch(acc, acc_frac, exact)
    assert (frac, worst) == (0.0, 0), f"{activation} should be exact at full resolution"

    # one bit of index resolution short - no longer exact
    coarse = ActLut.for_bundle(activation, in_bits=acc_frac + domain_bits,
                               in_frac=acc_frac - 1, **out)
    frac_coarse, _ = mismatch(acc, acc_frac, coarse)
    assert frac_coarse > 0.0, f"{activation} should not be exact below full resolution"


def test_mismatch_detects_a_too_coarse_table():
    # Same output grid, but the table only distinguishes 16 input buckets - the
    # accumulator carries far more information than that, so disagreement is
    # expected and must be reported rather than silently tolerated.
    coarse = ActLut('silu', in_bits=4, in_frac=1, in_signed=True,
                    out_bits=8, out_frac=4, out_signed=True)
    acc = np.arange(-512, 512, dtype=np.int64)
    frac, worst = mismatch(acc, 6, coarse)
    assert frac > 0.5 and worst > 0


# ---- float precision must match brevitas, not exceed it ----

def test_table_is_built_in_float32_like_brevitas():
    """The table must reproduce brevitas, which evaluates activations in float32
    - so computing it more precisely is a bug, not an improvement.

    silu(24.75) is the concrete case that exposed this: 24.75 exactly in float32,
    but 24.749999999558646 in float64. At out_frac=1 those round to 50 and 49,
    so a float64-built table disagrees with brevitas by 1 LSB on that entry.

    It only bites once activations get large, which is why neither XOR nor a
    standardised regression target surfaced it - it took the letter-recognition
    benchmark, where 10-bit tables showed 0.2% disagreement against brevitas."""
    lut = ActLut('silu', in_bits=10, in_frac=3, in_signed=True,
                 out_bits=8, out_frac=1, out_signed=True)
    # index 198 is the level 198/2**3 = 24.75
    assert lut.lookup(np.int64(198)) == 50, (
        "table entry for silu(24.75) is not brevitas's float32 answer - the "
        "table is probably being built in float64")


def test_exact_activation_shares_the_tables_precision():
    """exact_activation models brevitas, so it has to round the same way the
    table does. If these two ever disagree, `mismatch()` reports differences
    that are artefacts of its own arithmetic rather than real ones."""
    acc = np.array([24.75 * 2 ** 8], dtype=np.int64)
    ref = exact_activation(acc, 8, 'silu', out_bits=8, out_frac=1, out_signed=True)
    assert ref[0] == 50


@pytest.mark.parametrize("activation", sorted(CURVED_ACTIVATIONS))
def test_large_inputs_agree_with_float32_reference(activation):
    """Generalises the silu(24.75) case: across a range large enough for
    float32/float64 to diverge, the table and the reference must still agree
    everywhere."""
    lut = ActLut(activation, in_bits=12, in_frac=4, in_signed=True,
                 out_bits=8, out_frac=1,
                 out_signed=activation != 'sigmoid')
    acc = np.arange(-2000, 2000, dtype=np.int64) * 16  # acc_frac 8, wide range
    frac, worst = mismatch(acc, 8, lut)
    assert (frac, worst) == (0.0, 0), f"{activation}: {frac:.4%} disagreement"
