"""Adapts brevitas FixedPointModel bundles onto the legacy XBundle attribute
surface, so the legacy engine-layout export (deepsocflow/py/xmodel.py) and RTL
verification path run over brevitas-produced numbers unchanged.

brevitas owns the numbers; the legacy backend owns the file format. This module
is the only seam between them."""
import math
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
        log_slope = math.log2(negative_slope)
        assert log_slope == int(log_slope) and log_slope <= 0, (
            f"negative_slope={negative_slope} must be a negative power of two "
            f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")
        return 1, -int(log_slope)
    raise NotImplementedError(
        f"activation '{activation}' has no integer-exact hardware implementation "
        f"(see CLAUDE.md Known Issues); only relu/identity/leaky_relu are deployable")


def to_engine_weight(weight_int):
    """torch Linear weight (out_features, in_features) -> legacy conv weight
    (KH, KW, CI, CO) = (1, 1, in_features, out_features).

    The transpose is real: torch stores (out, in), keras stores (in, out)."""
    return np.asarray(weight_int).T[None, None, :, :]


def to_engine_activation(x_int):
    """(batch, features) -> (XN, XH, XW, CI) = (1, batch, 1, features).

    Batch goes in the H slot, not the N slot - this mirrors the legacy dense
    reshape at xbundle.py:126. Getting it backwards produces wrong runtime
    params (XL, X_PAD) without any error."""
    return np.asarray(x_int)[None, :, None, :]
