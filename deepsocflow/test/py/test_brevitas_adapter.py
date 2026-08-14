import re

import numpy as np
import pytest

from deepsocflow.py.brevitas.export.adapter import act_params


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


def test_act_params_rejects_default_slope_with_legible_message():
    # negative_slope defaults to 0.0, which used to reach math.log2(0.0) and
    # raise "ValueError: math domain error" before the power-of-two assertion
    # could produce its message (legacy's np.log2 degrades to -inf instead and
    # lets the assert fire cleanly). This is the only power-of-two enforcement
    # in the pipeline, so calling act_params('leaky_relu') with no slope must
    # raise the same legible AssertionError, not a ValueError.
    with pytest.raises(AssertionError, match="power of two"):
        act_params('leaky_relu')


def test_act_params_rejects_unsupported_activation():
    with pytest.raises(NotImplementedError, match="silu"):
        act_params('silu')


import json

from deepsocflow.py.brevitas.simulation.sim import FixedPointModel


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
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    assert [b.ib for b in bundles] == [0, 1]
    assert bundles[0].prev_ib is None
    assert bundles[1].prev_ib == 0
    assert sorted(bundles[0].next_ibs) == [1]
    assert sorted(bundles[1].next_ibs) == []


def test_build_bundles_registers_into_legacy_bundles_global(tmp_path):
    from deepsocflow.py.numeric import BUNDLES
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2

    # building again must not accumulate
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2


def test_build_bundles_shift_bits_matches_sim(tmp_path):
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    # acc_frac = input_frac + weight_frac = 7 + 6 = 13; act_frac = 6
    # shift_bits = plog_slope + acc_frac - act_frac = 0 + 13 - 6 = 7
    assert bundles[0].core.act.shift_bits == 7
    assert bundles[0].core.act.non_zero == 0      # relu
    assert bundles[1].core.act.non_zero == 1      # identity


def test_call_int_is_a_noop(tmp_path):
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    before = bundles[0].core.y.itensor.copy()
    bundles[0].call_int(None, hw)
    assert np.array_equal(bundles[0].core.y.itensor, before)


def test_bias_none_when_absent(tmp_path):
    """legacy xbundle.py:135,167 tests `if self.core.b` truthiness - an absent
    bias must be None, never an empty/zero array."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

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


def test_to_legacy_dense_weight_transposes_to_in_out():
    """torch stores a Linear weight as (out, in); legacy's dense branch expects
    (CI, CO) = (in, out) and does the reshape to (1,1,CI,CO) itself."""
    from deepsocflow.py.brevitas.export.adapter import to_legacy_dense_weight

    w = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])          # (out=3, in=2)
    legacy = to_legacy_dense_weight(w)
    assert legacy.shape == (2, 3)
    assert legacy[0, 1] == 3        # element (in=0, out=1) is w[out=1][in=0]
    assert legacy[1, 2] == 6


def test_core_tensors_are_2d_for_legacy_dense_branch(tmp_path):
    """xbundle.py:139 does `CI,CO = core.w.itensor.shape` — a 4-D tensor here
    raises 'too many values to unpack'."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    for b in bundles:
        assert len(b.core.w.itensor.shape) == 2, "weight must stay (CI, CO)"
        assert len(b.core.x.itensor.shape) == 2, "input must stay (XN, CI)"
        assert len(b.core.y.itensor.shape) == 2, "conv-sum must stay (XN, CO)"
    assert bundles[-1].pre_softmax is not None
    assert len(bundles[-1].pre_softmax.itensor.shape) == 2


def test_core_exposes_bias_shifts(tmp_path):
    """xmodel.py:234's config_fw.h writer reads these. Legacy computes them in
    XDense.call_int, which the adapter's no-op call_int never runs."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    for b in bundles:
        assert b.core.bias_val_shift == 0
        assert b.core.bias_b_shift == 0


def test_legacy_xbundle_export_runs_on_adapted_bundles(tmp_path):
    """The gap that let both defects through: no Task 6 test called .export().
    This drives the real legacy reorder path end to end."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    for i, b in enumerate(bundles):
        b.export(hw, is_last=(i == len(bundles) - 1))

    for b in bundles:
        assert b.we is not None and len(b.we) > 0
        assert b.xe is not None and len(b.xe) > 0
        assert len(b.ye_exp_p) == b.r.CP
        assert b.oe_exp_nhwc is not None


def test_softmax_fields_default_to_zero_and_are_set_on_the_softmax_bundle(tmp_path):
    """xmodel.py:234 reads b.softmax_frac and b.softmax_max_i. Legacy defaults
    both to 0 (xbundle.py:47-48) and overrides them only on a softmax bundle."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    assert bundles[0].softmax_frac == 0
    assert bundles[0].softmax_max_i == 0

    # bundle 1 is the softmax bundle in this fixture
    assert bundles[1].softmax_frac == 6          # its act_frac
    assert isinstance(bundles[1].softmax_max_i, int), \
        "config_fw.h has one scalar field per bundle; a per-row array cannot go in it"


def test_full_legacy_export_path_runs_on_adapted_bundles(tmp_path, monkeypatch):
    """Drives _export_bundles - the same function the real driver calls, and the
    one that writes config_fw.h. .export() alone is only half the path: every
    defect so far has been an attribute legacy sets inside call_int, which the
    adapter no-ops, and several are read only by the config_fw.h writer."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    build_bundles(_two_bundle_model(tmp_path), hw)

    monkeypatch.chdir(tmp_path)   # config_fw.h is written to the CWD, not DATA_DIR
    _export_bundles(hw, None)     # x=None: the adapter's call_int is a no-op

    assert (tmp_path / "config_fw.h").exists(), "config_fw.h was not written"
    assert any(data_dir.iterdir()), "no engine-layout files were written"


def test_config_fw_h_flags_flatten_and_softmax_correctly(tmp_path, monkeypatch):
    """xmodel.py:233 emits these with `is not None`, and legacy stores None when
    absent (xbundle.py:41,44). Storing False instead makes every bundle claim to
    be flattened and softmaxed. Asserting on the emitted text rather than on the
    attributes is deliberate: that is the level this bug is visible at."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    build_bundles(_two_bundle_model(tmp_path), hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    text = (tmp_path / "config_fw.h").read_text()
    flatten_flags = [int(v) for v in re.findall(r"\.is_flatten=\s*(\d+)", text)]
    softmax_flags = [int(v) for v in re.findall(r"\.is_softmax=\s*(\d+)", text)]

    assert flatten_flags == [0, 0], "no bundle in this fixture is flattened"
    assert softmax_flags == [0, 1], "only the last bundle carries softmax"


def test_absent_flatten_and_softmax_are_none_not_false(tmp_path):
    """Legacy stores None; xmodel.py:233 tests `is not None` while xbundle.py:144
    tests truthiness, so absent must be None and present must be truthy."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    for b in bundles:
        assert b.flatten is None
    assert bundles[0].softmax is None
    assert bundles[1].softmax
