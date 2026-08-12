"""Value-LUT activations for the brevitas fixed-point path.

Why this exists
---------------
`quant_lrelu` (deepsocflow/c/runtime.h:153) can only realize activations that are
piecewise-linear through the origin (relu / power-of-two-slope leaky_relu /
identity), because it computes the whole activation as shift + clip. Curved
activations - silu, tanh, sigmoid, gelu, selu - have no such closed form, which
is why `sim.py`'s SUPPORTED_ACTIVATIONS has been limited to relu/identity.

A value LUT closes that gap without adding any arithmetic: the activation
becomes a table load, so the "no multiplier, shift only" property of every other
rescale in this project is preserved exactly. Prior art is hls4ml's `UnaryLUT`
(hls4ml/converters/keras_v3/hgq2/unary_lut.py).

Where this sits in the datapath
-------------------------------
Activation is computed on the host CPU (`runtime.h`), not in RTL - the RTL is a
MAC array that writes raw i32 accumulators to OCM and nothing else (verified:
no activation logic under deepsocflow/rtl/, and `create_headers` in
deepsocflow/py/dataflow.py packs only tiling geometry into the header the RTL
receives). So a LUT activation is a firmware + config change; it needs no RTL
edit and no bitstream rebuild.

The index is NOT computed here
------------------------------
This module only owns the table. Turning an accumulator into a table index is
the caller's job, because it is the shift the pipeline already performs:

    idx = clip(shift_round(acc, acc_frac - lut.in_frac), lut.in_bits)   # caller
    out = lut.lookup(idx)                                               # here

which mirrors what the C firmware would do:

    x = shift_round(x, shift);
    x = clip(x, ...);
    return lut[(u8)x];

`sim.py::FixedPointModel.forward` is the caller today. Keeping the split here
means the shift stays in one place and this module never needs `shift_round`.

Raw two's-complement indexing
-----------------------------
The table is stored pre-permuted - entries for non-negative levels first, then
entries for negative levels - so a signed index can address it by its raw
two's-complement bit pattern (`idx & (2**in_bits - 1)`) with no `+2**(bits-1)`
bias add. Same trick hls4ml's UnaryLUT uses, and it costs the firmware nothing
but a cast.
"""

import math

import numpy as np

# Float reference implementations, keyed by the activation names
# ptq.py::_act_type_name emits into the graph JSON.
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

_erf = np.vectorize(math.erf)

ACT_FUNCS = {
    'identity': lambda x: x,
    'relu': lambda x: np.maximum(x, 0.0),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
    'tanh': np.tanh,
    'silu': lambda x: x / (1.0 + np.exp(-x)),
    'gelu': lambda x: 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0))),
    'selu': lambda x: _SELU_SCALE * np.where(x > 0, x, _SELU_ALPHA * (np.expm1(x))),
}

# Activations with no closed-form shift-and-clip implementation. These are the
# ones a LUT actually buys us; relu/identity/leaky_relu stay on the cheaper
# quant_lrelu path (a LUT would work for them too, it would just be wasteful).
CURVED_ACTIVATIONS = frozenset({'sigmoid', 'tanh', 'silu', 'gelu', 'selu'})


def levels(bits, signed):
    """The integer levels an `bits`-wide fixed-point word can hold, in raw
    two's-complement index order: non-negative levels first, then negative ones.

    Indexing the returned array by `value & (2**bits - 1)` therefore yields
    `value`, which is what lets the table be addressed by raw bits."""
    raw = np.arange(2 ** bits, dtype=np.int64)
    if not signed:
        return raw
    return np.where(raw < 2 ** (bits - 1), raw, raw - 2 ** bits)


def clip_to(values, bits, signed):
    """Saturating clip to a fixed-point word, matching how brevitas's own
    quantizers clip rather than wrap."""
    if signed:
        return np.clip(values, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return np.clip(values, 0, 2 ** bits - 1)


class ActLut:
    """A precomputed activation table plus the fixed-point grids on either side
    of it.

    `in_bits`/`in_frac` describe the table's *index* grid (how finely and over
    what range the accumulator is bucketed before lookup); `out_bits`/`out_frac`
    describe the activation output, and are fixed by the model's own act_quant.

    **These two widths are independent, and only `out_bits` is a hardware
    constraint.** The index never leaves the CPU - it is computed in an i32
    register, used once to address the table, and discarded; only the table's
    *entries* are stored as activation words and therefore have to fit
    `hw.X_BITS`. So an 8-bit activation datapath happily runs a 10- or 12-bit
    index: the table just has more rows, each still one byte. `check_hardware`
    (export.py) enforces exactly that asymmetry - `out_bits <= hw.X_BITS`, while
    `in_bits` is bounded only by table size. Verified end to end: a 10-bit-index
    model on `bits_input=8` hardware exports `X_BITS=8` with `LUT_ENTRIES 1024`
    and passes RTL simulation with `Error: 0`.

    The index grid is deliberately a free parameter rather than being tied to
    the output grid - see `for_bundle` for why that matters."""

    def __init__(self, activation, in_bits, in_frac, in_signed,
                 out_bits, out_frac, out_signed, negative_slope=None):
        if activation not in ACT_FUNCS and activation != 'leaky_relu':
            raise ValueError(
                f"no float reference for activation '{activation}' - "
                f"known: {sorted(ACT_FUNCS) + ['leaky_relu']}")
        for name, value in (('in_bits', in_bits), ('in_frac', in_frac),
                            ('out_bits', out_bits), ('out_frac', out_frac)):
            if int(value) != value:
                raise ValueError(f"{name}={value} must be an integer - every scale in "
                                 f"this pipeline is a power of two (2**-frac)")
        if in_bits > 24:
            raise ValueError(
                f"in_bits={in_bits} would need a {2 ** in_bits} entry table - refusing "
                f"to build. Curved activations saturate long before this helps.")

        self.activation = activation
        self.in_bits, self.in_frac, self.in_signed = int(in_bits), int(in_frac), bool(in_signed)
        self.out_bits, self.out_frac, self.out_signed = int(out_bits), int(out_frac), bool(out_signed)
        self.negative_slope = negative_slope

        if activation == 'leaky_relu':
            slope = negative_slope
            log_slope = math.log2(slope) if slope and slope > 0 else float('-inf')
            # Mirrors ptq.py::_quantize_activation and adapter.py::act_params: a
            # leaky_relu whose slope is not a negative power of two has no valid
            # quant_lrelu parameterization. A LUT could technically represent any
            # slope, but allowing one here would let a model through that the
            # rest of the pipeline still cannot deploy.
            if not (slope > 0 and int(log_slope) == log_slope and log_slope <= 0):
                raise ValueError(
                    f"negative_slope={slope} must be a negative power of two "
                    f"(0.5, 0.25, 0.125, ...)")
            fn = lambda x: np.where(x >= 0, x, x * slope)
        else:
            fn = ACT_FUNCS[activation]

        # float32, NOT float64 - this is load-bearing. brevitas evaluates
        # activations in float32, and for large inputs the two precisions
        # disagree across a rounding boundary: silu(24.75) is 24.75 exactly in
        # float32 but 24.749999999558646 in float64, so at out_frac=1 they round
        # to 50 and 49 respectively. Building the table in float64 therefore
        # makes it *more* mathematically accurate and *less* faithful to the
        # model it has to reproduce - and only on inputs large enough for the
        # deviation to matter, which is why XOR and a normalised regression
        # target never surfaced it. Found via the letter-recognition benchmark,
        # where 10-bit tables showed 0.2% disagreement.
        idx_levels = levels(self.in_bits, self.in_signed)
        x_float = (idx_levels.astype(np.float32)
                   / np.float32(2.0 ** self.in_frac)).astype(np.float32)
        y_float = np.asarray(fn(x_float), dtype=np.float32)
        # np.rint is round-half-to-even, the same tie-breaking shift_round and
        # brevitas's own quantizers use - so the table agrees with the rest of
        # the pipeline on the boundary cases, not just the easy ones.
        y_int = np.rint((y_float * np.float32(2.0 ** self.out_frac)).astype(np.float64))
        self.table = clip_to(y_int, self.out_bits, self.out_signed).astype(np.int64)

    # ---- construction from a bundle's own quantization params ----

    @classmethod
    def for_bundle(cls, activation, act_bits, act_frac, act_signed,
                   in_bits=None, in_frac=None, in_signed=True, negative_slope=None):
        """Build the table for a bundle whose activation output is
        (`act_bits`, `act_frac`, `act_signed`).

        With `in_bits`/`in_frac` left as None this is variant "1a": index the
        table with the value the pipeline already computes, i.e. the accumulator
        rescaled onto the *output* grid. That needs no new quantization point in
        brevitas at all - it reuses the shift `forward()` already performs.

        Tying the index grid to the output grid is only sound when the two
        cover comparable ranges. It holds for silu (asymptotically identity, so
        input and output ranges nearly coincide) and fails badly for the
        saturating functions - tanh/sigmoid outputs live in [-1,1], so their
        act_frac is large and the resulting index range is far too narrow to
        cover the input domain those functions actually need. Pass `in_bits`/
        `in_frac` explicitly for those. Note that doing so is still not a
        brevitas quantization point: it only changes how the accumulator is
        bucketed inside this activation, which is invisible to the graph."""
        return cls(
            activation=activation,
            in_bits=act_bits if in_bits is None else in_bits,
            in_frac=act_frac if in_frac is None else in_frac,
            in_signed=in_signed,
            out_bits=act_bits, out_frac=act_frac, out_signed=act_signed,
            negative_slope=negative_slope,
        )

    # ---- use ----

    def index_shift(self, acc_frac):
        """The right-shift that moves an accumulator at `acc_frac` onto this
        table's index grid.

        Asserting this is non-negative is the power-of-two guard: every scale in
        this pipeline is 2**-frac, so the rescale is a pure shift and never a
        multiply. A negative result would mean the table is finer than the
        accumulator itself, which is never useful (it would interpolate
        precision that does not exist) and would need a left shift the firmware
        does not do."""
        shift = acc_frac - self.in_frac
        if shift < 0:
            raise ValueError(
                f"acc_frac={acc_frac} is coarser than the table's in_frac={self.in_frac} - "
                f"the index rescale must be a right shift (acc_frac >= in_frac)")
        return shift

    def lookup(self, idx_int):
        """Table lookup by raw two's-complement bits.

        `idx_int` must already be shifted onto the index grid and clipped to
        `in_bits` - see this module's docstring for why the caller owns that."""
        idx = np.asarray(idx_int, dtype=np.int64) & (2 ** self.in_bits - 1)
        return self.table[idx]

    # ---- reporting ----

    @property
    def nbytes(self):
        """Bytes the table costs on the device, at the output word width."""
        return int(2 ** self.in_bits * math.ceil(self.out_bits / 8))

    @property
    def input_range(self):
        """(lo, hi) real-valued span this table can address. Accumulators outside
        it saturate to the end entries."""
        lo = -2.0 ** (self.in_bits - 1) if self.in_signed else 0.0
        hi = (2.0 ** (self.in_bits - 1) - 1) if self.in_signed else (2.0 ** self.in_bits - 1)
        return lo / 2.0 ** self.in_frac, hi / 2.0 ** self.in_frac

    def __repr__(self):
        lo, hi = self.input_range
        return (f"ActLut({self.activation}, index {self.in_bits}b/frac{self.in_frac} "
                f"covering [{lo:g}, {hi:g}], out {self.out_bits}b/frac{self.out_frac}, "
                f"{self.nbytes} B)")


# ---- reference / measurement ----

def exact_activation(acc, acc_frac, activation, out_bits, out_frac, out_signed,
                     negative_slope=None):
    """The activation computed at full accumulator precision, then quantized.

    This is what brevitas's own fake-quantized forward pass produces (it applies
    the float activation to the dequantized accumulator, then quantizes the
    result), so it is the reference a LUT has to match to be called bit-exact.
    A LUT reproduces it exactly only when its index grid is fine enough that no
    two accumulators sharing an index would have quantized to different
    outputs."""
    if activation == 'leaky_relu':
        fn = lambda x: np.where(x >= 0, x, x * negative_slope)
    else:
        fn = ACT_FUNCS[activation]
    # float32 for the same reason ActLut's table is - see the comment there.
    # This function models brevitas, so it has to share brevitas's precision.
    x_float = (np.asarray(acc, dtype=np.float32)
               / np.float32(2.0 ** acc_frac)).astype(np.float32)
    y_float = np.asarray(fn(x_float), dtype=np.float32)
    y_int = np.rint((y_float * np.float32(2.0 ** out_frac)).astype(np.float64))
    return clip_to(y_int, out_bits, out_signed).astype(np.int64)


def lut_activation(acc, acc_frac, lut):
    """Apply `lut` to accumulators - the same three steps `sim.py` performs, kept
    here so measurement code and tests can call the pipeline in one line."""
    from deepsocflow.py.brevitas.sim import shift_round
    idx = shift_round(acc, lut.index_shift(acc_frac))
    idx = clip_to(idx, lut.in_bits, lut.in_signed)
    return lut.lookup(idx)


def mismatch(acc, acc_frac, lut, negative_slope=None):
    """Fraction of `acc` values where the LUT disagrees with `exact_activation`,
    plus the largest disagreement in output LSBs."""
    ref = exact_activation(acc, acc_frac, lut.activation, lut.out_bits,
                           lut.out_frac, lut.out_signed, negative_slope)
    got = lut_activation(acc, acc_frac, lut)
    diff = np.abs(got - ref)
    return float((diff != 0).mean()), int(diff.max()) if diff.size else 0
