import json

import numpy as np
import pytest

from deepsocflow.py.brevitas import sim
from deepsocflow.py.brevitas.sim import FixedPointModel


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


def test_requantizes_between_bundles_when_input_frac_differs_from_prev_act_frac(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
        "bundle1": _bundle(input_frac=5, input_bits=8,
                            weight_values=[[32]], weight_frac=5, weight_bits=8,
                            bias_values=[0], bias_frac=10, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=5,
                            input_name="bundle0"),
    }
    json_path = _write_graph(tmp_path, layers)
    model = sim.FixedPointModel(json_path)
    model.load_int_weights(json_path)

    # bundle0: x=[0.5,0.5] (int [64,64] @ frac=7), weight=[1.0,1.0] (int [64,64] @ frac=6)
    #   -> acc = 64*64 + 64*64 = 8192 @ frac=13 -> shift_round(8192, 7) = 64 @ frac=6 (=1.0)
    # bundle1 expects its input at frac=5, but bundle0's output is at frac=6 - must
    # requantize: shift_round(64, 6-5=1) = 32 @ frac=5 (still =1.0) before the matmul.
    out = model.forward(np.array([[64, 64]], dtype=np.int64))

    assert out.tolist() == [[32]]


def test_forward_records_trace_per_bundle(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
        "bundle1": _bundle(input_frac=5, input_bits=8,
                            weight_values=[[32]], weight_frac=5, weight_bits=8,
                            bias_values=[0], bias_frac=10, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=5,
                            input_name="bundle0"),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    model.forward(np.array([[64, 64]], dtype=np.int64))

    assert set(model.trace.keys()) == {"bundle0", "bundle1"}
    for name in model.trace:
        t = model.trace[name]
        assert np.array_equal(t["acc"], t["y"] + model.bundles[name]["bias"])


def test_forward_before_load_int_weights_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)  # load_int_weights() deliberately NOT called

    with pytest.raises(RuntimeError, match="load_int_weights"):
        model.forward(np.array([[64, 64]], dtype=np.int64))


def test_bias_less_layer_defaults_to_zeros(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    out = model.forward(np.array([[64, 64]], dtype=np.int64))

    # y = 64*64 + 64*64 = 8192 @ frac=13, bias=0 -> acc=8192
    # shift_round(8192, 13-6=7) = 64 @ frac=6 (=1.0)
    assert out.tolist() == [[64]]
    assert model.trace["bundle0"]["acc"].tolist() == [[8192]]


def test_softmax_output_matches_manual_softmax(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0, 0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6, softmax=True),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    logits_int = model.forward(np.array([[64, 0]], dtype=np.int64))

    # weight is the identity matrix (scaled by 2**6), so bundle0's logits are
    # exactly the (rescaled) input: x=[64,0] @ frac=7 (=[0.5,0.0]) times identity
    # -> acc=[4096,0] @ frac=13 -> shift_round(.,7) -> int [32,0] @ act_frac=6
    assert logits_int.tolist() == [[32, 0]]
    assert model.pre_softmax.tolist() == logits_int.tolist()
    assert model.softmax_frac == 6

    logits_float = np.array([32, 0]) / 2 ** 6  # == [0.5, 0.0]
    expected = np.exp(logits_float) / np.exp(logits_float).sum()
    assert model.softmax_out[0].tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_unsupported_activation_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="silu", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    with pytest.raises(ValueError, match="silu"):
        FixedPointModel(json_path)


def test_unsupported_layer_type_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6, type_="conv"),
    }
    json_path = _write_graph(tmp_path, layers)
    with pytest.raises(ValueError, match="conv"):
        FixedPointModel(json_path)


def test_prefers_act_signed_field_over_name_based_fallback(tmp_path):
    # A relu bundle whose JSON explicitly marks act_signed=True (e.g. hand-edited,
    # or from a future export where ReLU's output was clipped signed) must be
    # treated as signed even though 'relu' is in UNSIGNED_ACTIVATIONS by name.
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[127, 127]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="relu", act_bits=8, act_frac=6),
    }
    layers["bundle0"]["act_signed"] = True
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    # x=[127,127] @ frac=7, weight=[127,127] @ frac=6 -> acc=127*127+127*127=32258
    # @ frac=13 -> relu(32258)=32258 (already >=0) -> shift_round(32258, 7) = 252
    # pre-clip. Signed int8 clips this to 127; unsigned uint8 would leave it at
    # 252 - this is the discriminating case between the two range behaviors.
    out = model.forward(np.array([[127, 127]], dtype=np.int64))
    assert out.tolist() == [[127]]  # signed range [-128,127], not unsigned [0,255]


def test_json_has_single_quantization_point_per_bundle():
    """Regenerates the real XOR graph and checks every bundle after the first
    has input_frac == the previous bundle's act_frac - i.e. Task 3's requant
    step is now a structural no-op, not a band-aid."""
    import json
    import subprocess
    import sys

    # NOTE: deliberately uses this worktree's own checkout, not the brief's literal
    # "/Users/charaphat/CERN/cgra4ml" - this task must run and be verified inside the
    # brevitas-golden-ref worktree (a separate checkout on its own branch), and the
    # main repo path points at an unrelated, actively-edited branch
    # (brevitas-qonnx-backend) whose ptq.py/xor.py have diverged - running against it
    # would test the wrong code and risk interfering with that other work.
    repo_root = "/Users/charaphat/CERN/cgra4ml/.worktrees/brevitas-golden-ref"
    subprocess.run([sys.executable, "-m", "deepsocflow.py.brevitas.xor"], check=True,
                    cwd=repo_root, capture_output=True)

    with open(f"{repo_root}/deepsocflow/py/brevitas/model/xor_graph.json") as f:
        layers = json.load(f)["layers"]

    names = list(layers.keys())
    for i in range(1, len(names)):
        prev, cur = layers[names[i - 1]], layers[names[i]]
        assert cur["input_frac"] == prev["act_frac"], (
            f"{names[i]}.input_frac ({cur['input_frac']}) != "
            f"{names[i-1]}.act_frac ({prev['act_frac']})")
