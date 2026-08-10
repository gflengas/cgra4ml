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
