import numpy as np
import pytest

from deepsocflow.py.brevitas.adapter import act_params


def test_act_params_relu():
    # legacy: slope=0 -> non_zero = 1*(0 != 0) = 0, plog_slope = 0
    assert act_params('relu') == (0, 0)


def test_act_params_identity():
    # legacy: type=None forces slope=1 -> non_zero = 1, log2(1) = 0
    assert act_params('identity') == (1, 0)


def test_act_params_leaky_relu_power_of_two():
    assert act_params('leaky_relu', negative_slope=0.125) == (1, 3)
    assert act_params('leaky_relu', negative_slope=0.5) == (1, 1)


def test_act_params_rejects_non_power_of_two_slope():
    with pytest.raises(AssertionError, match="power of two"):
        act_params('leaky_relu', negative_slope=0.1)


def test_act_params_rejects_unsupported_activation():
    with pytest.raises(NotImplementedError, match="silu"):
        act_params('silu')


from deepsocflow.py.brevitas.adapter import to_engine_activation, to_engine_weight


def test_to_engine_weight_shape_and_transpose():
    # torch Linear weight is (out_features, in_features); keras/legacy wants
    # (KH, KW, CI, CO) = (1, 1, in_features, out_features)
    w = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])          # (out=3, in=2)
    e = to_engine_weight(w)
    assert e.shape == (1, 1, 2, 3)
    # element (in=0, out=1) must be w[out=1][in=0] == 3
    assert e[0, 0, 0, 1] == 3
    assert e[0, 0, 1, 2] == 6


def test_to_engine_activation_puts_batch_in_h_slot():
    # (batch, features) -> (XN, XH, XW, CI) = (1, batch, 1, features).
    # Batch lands in H, NOT in N - see xbundle.py:126.
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])          # (batch=4, features=2)
    e = to_engine_activation(x)
    assert e.shape == (1, 4, 1, 2)
    assert e[0, 2, 0, 0] == 1       # row 2 is [1, 0]
    assert e[0, 2, 0, 1] == 0


def test_to_engine_roundtrip_preserves_values():
    x = np.arange(12).reshape(4, 3)
    assert to_engine_activation(x).flatten().tolist() == x.flatten().tolist()
