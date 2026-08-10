import json

import numpy as np
import pytest

from deepsocflow.py.brevitas import sim


def _bundle(input_frac, input_bits, weight_values, weight_frac, weight_bits,
            activation, act_bits, act_frac, bias_values=None, bias_frac=None,
            bias_bits=None, softmax=False, input_name=None, type_="linear"):
    """Builds one entry of a graph JSON's "layers" dict, matching the schema
    quantized_model.export_graph_json() produces (deepsocflow/py/brevitas/ptq.py).
    Only includes the fields sim.py actually reads."""
    in_features = len(weight_values[0])
    out_features = len(weight_values)
    layer = {
        "type": type_,
        "input": input_name,
        "input_bits": input_bits,
        "input_frac": input_frac,
        "in_features": in_features,
        "out_features": out_features,
        "weight": {"bits": weight_bits, "frac": weight_frac, "values": weight_values},
        "activation": activation,
        "act_bits": act_bits,
        "act_frac": act_frac,
        "softmax": softmax,
    }
    if bias_values is not None:
        layer["bias"] = {"bits": bias_bits, "frac": bias_frac, "values": bias_values}
    return layer


def _write_graph(tmp_path, layers, name="graph.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"layers": layers}))
    return str(path)


def test_shift_round_matches_legacy():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.utils import shift_round as legacy_shift_round

    rng = np.random.default_rng(0)
    n = rng.integers(-100000, 100000, size=2000, dtype=np.int64)
    for s in range(1, 13):
        ours = sim.shift_round(n, s)
        legacy = np.asarray(legacy_shift_round(n, s), dtype=np.int64)
        assert np.array_equal(ours, legacy), f"mismatch at s={s}"


def test_shift_round_rounds_half_to_even():
    n = np.array([-10, -6, -2, 2, 6, 10], dtype=np.int64)
    # dividing by 4 (s=2): exact-half results round to the nearest EVEN value
    assert sim.shift_round(n, 2).tolist() == [-2, -2, 0, 0, 2, 2]


def test_quantize_input_clips_to_input_bits(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = sim.FixedPointModel(json_path)

    x_int = model.quantize_input([[1.0, 0.0]])

    # 1.0 * 2**7 = 128, but signed int8 tops out at 127 - must clip, not wrap.
    assert x_int.tolist() == [[127, 0]]
