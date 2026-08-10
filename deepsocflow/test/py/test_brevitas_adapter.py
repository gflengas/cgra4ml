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


import json

from deepsocflow.py.brevitas.sim import FixedPointModel


def _bundle_cfg(input_frac, input_bits, weight_values, weight_frac, weight_bits,
                activation, act_bits, act_frac, bias_values, bias_frac, bias_bits,
                softmax=False, input_name=None):
    return {
        "type": "linear",
        "input": input_name,
        "input_bits": input_bits,
        "input_frac": input_frac,
        "input_signed": True,
        "in_features": len(weight_values[0]),
        "out_features": len(weight_values),
        "weight": {"bits": weight_bits, "frac": weight_frac, "values": weight_values},
        "bias": {"bits": bias_bits, "frac": bias_frac, "values": bias_values},
        "activation": activation,
        "act_bits": act_bits,
        "act_frac": act_frac,
        "act_signed": activation != 'relu',
        "softmax": softmax,
    }


def _two_bundle_model(tmp_path):
    """A 2-input -> 2-hidden (relu) -> 2-output (identity+softmax) chain, run
    forward so .trace is populated."""
    layers = {
        "bundle0": _bundle_cfg(
            input_frac=7, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=13, bias_bits=16,
            activation="relu", act_bits=8, act_frac=6),
        "bundle1": _bundle_cfg(
            input_frac=6, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=12, bias_bits=16,
            activation="identity", act_bits=8, act_frac=6,
            softmax=True, input_name="bundle0"),
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"layers": layers}))
    model = FixedPointModel(str(path))
    model.load_int_weights(str(path))
    model.forward(model.quantize_input([[1.0, 0.0], [0.0, 1.0]]))
    return model


def test_build_bundles_sets_chain_topology(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    assert [b.ib for b in bundles] == [0, 1]
    assert bundles[0].prev_ib is None
    assert bundles[1].prev_ib == 0
    assert sorted(bundles[0].next_ibs) == [1]
    assert sorted(bundles[1].next_ibs) == []


def test_build_bundles_registers_into_legacy_bundles_global(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.utils import BUNDLES
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2

    # building again must not accumulate
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2


def test_build_bundles_shift_bits_matches_sim(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    # acc_frac = input_frac + weight_frac = 7 + 6 = 13; act_frac = 6
    # shift_bits = plog_slope + acc_frac - act_frac = 0 + 13 - 6 = 7
    assert bundles[0].core.act.shift_bits == 7
    assert bundles[0].core.act.non_zero == 0      # relu
    assert bundles[1].core.act.non_zero == 1      # identity


def test_call_int_is_a_noop(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    before = bundles[0].core.y.itensor.numpy().copy()
    bundles[0].call_int(None, hw)
    assert np.array_equal(bundles[0].core.y.itensor.numpy(), before)


def test_bias_none_when_absent(tmp_path):
    """legacy xbundle.py:135,167 tests `if self.core.b` truthiness - an absent
    bias must be None, never an empty/zero array."""
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    layers = {
        "bundle0": _bundle_cfg(
            input_frac=7, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=13, bias_bits=16,
            activation="identity", act_bits=8, act_frac=6),
    }
    del layers["bundle0"]["bias"]
    path = tmp_path / "nobias.json"
    path.write_text(json.dumps({"layers": layers}))
    model = FixedPointModel(str(path))
    model.load_int_weights(str(path))
    model.forward(model.quantize_input([[1.0, 0.0]]))

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(model, hw, has_bias={"bundle0": False})
    assert bundles[0].core.b is None
