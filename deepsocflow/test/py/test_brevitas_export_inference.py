import json
import os

import numpy as np
import pytest
import torch

from deepsocflow.py.brevitas.export.export import check_hardware, export_inference
from deepsocflow.py.brevitas.hardware.hardware import Hardware
from deepsocflow.py.brevitas.simulation.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import MODEL_DIR, MODEL_PATH, X, Y

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"'{MODEL_PATH}' not found - run `python -m deepsocflow.py.brevitas.xor` to generate it")


def _build_model(tmp_path):
    """Rebuilds quantized_model from the checked-in trained weights, regenerates
    the graph JSON into tmp_path (never reads the repo copy - it's a derived
    artefact and could be stale), and returns a forward()-ready FixedPointModel."""
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.xor import XOR, load

    net = XOR()
    net = load(net, path=MODEL_PATH)
    qm = quantized_model(net, layer_bits={
        'hidden_1': {'weight_bits': 8, 'bias_bits': 16},
        'hidden_2': {'weight_bits': 8, 'bias_bits': 16},
        'out': {'weight_bits': 8, 'bias_bits': 16},
    })
    qm.quantization(X)

    json_path = str(tmp_path / "xor_graph.json")
    qm.export_graph_json(X, json_path)

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    return model, qm


def _hardware(tmp_path):
    return Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        data_dir=str(tmp_path / "vectors"))


def test_export_inference_writes_y_exp_and_per_bundle_files(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)

    result = export_inference(model, hw, X, batch_size=1)

    assert os.path.exists(os.path.join(hw.DATA_DIR, "y_exp.txt"))
    for ib in range(len(model.bundle_order)):
        assert os.path.exists(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt"))
    assert set(result["files"]) == {
        os.path.join(hw.DATA_DIR, "y_exp.txt"),
        *(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt") for ib in range(len(model.bundle_order))),
    }


def test_y_exp_txt_matches_model_softmax_output(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    y_exp = np.loadtxt(os.path.join(hw.DATA_DIR, "y_exp.txt"))
    model.forward(model.quantize_input(X[:1]))
    assert y_exp.tolist() == pytest.approx(model.softmax_out[0].tolist(), abs=1e-6)


def test_y_exp_is_float_format_when_softmax(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    with open(os.path.join(hw.DATA_DIR, "y_exp.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert all("." in line for line in lines)


def test_y_nhwc_exp_is_int_format(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    with open(os.path.join(hw.DATA_DIR, "0_y_nhwc_exp.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        int(line)  # raises ValueError if not a plain int (e.g. "1.0" or "1e5")


def test_y_nhwc_exp_matches_bundle_trace_output(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    model.forward(model.quantize_input(X[:1]))
    for ib, name in enumerate(model.bundle_order):
        vals = np.loadtxt(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt"), dtype=np.int64)
        expected = model.trace[name]["out"].flatten()
        assert vals.tolist() == expected.tolist()


def test_export_cleans_data_dir(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    os.makedirs(hw.DATA_DIR, exist_ok=True)
    junk_path = os.path.join(hw.DATA_DIR, "leftover_junk.txt")
    with open(junk_path, 'w') as f:
        f.write("stale")

    export_inference(model, hw, X, batch_size=1)

    assert not os.path.exists(junk_path)


def test_hardware_bitwidth_mismatch_raises(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=4, bits_weights=8,
                   bits_bias=16, bits_sum=32, data_dir=str(tmp_path / "vectors"))

    with pytest.raises(AssertionError, match="X_BITS"):
        check_hardware(model, hw)


def test_hardware_weight_bits_mismatch_raises(tmp_path):
    # adapter.py's build_bundles labels every weight tensor with bits=hw.K_BITS
    # regardless of the JSON's real weight bit-width, so a mismatch here is
    # silent unless check_hardware catches it explicitly.
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.xor import XOR, load

    net = XOR()
    net = load(net, path=MODEL_PATH)
    qm = quantized_model(net, layer_bits={
        'hidden_1': {'weight_bits': 4, 'bias_bits': 16},
        'hidden_2': {'weight_bits': 8, 'bias_bits': 16},
        'out': {'weight_bits': 8, 'bias_bits': 16},
    })
    qm.quantization(X)
    json_path = str(tmp_path / "xor_graph.json")
    qm.export_graph_json(X, json_path)

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                   bits_bias=16, bits_sum=32, data_dir=str(tmp_path / "vectors"))
    model.forward(model.quantize_input(X[:1]))

    with pytest.raises(AssertionError, match="K_BITS"):
        check_hardware(model, hw)


def test_hardware_bias_bits_exceeding_b_bits_raises(tmp_path):
    # Failure scenario from the review: layer_bits sets bias_bits=32 while
    # Hardware only configures bits_bias=16 - xmodel.py's b.be.astype(np.int16)
    # would silently wrap those bias values into wb.bin without this check.
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.xor import XOR, load

    net = XOR()
    net = load(net, path=MODEL_PATH)
    qm = quantized_model(net, layer_bits={
        'hidden_1': {'weight_bits': 8, 'bias_bits': 32},
        'hidden_2': {'weight_bits': 8, 'bias_bits': 16},
        'out': {'weight_bits': 8, 'bias_bits': 16},
    })
    qm.quantization(X)
    json_path = str(tmp_path / "xor_graph.json")
    qm.export_graph_json(X, json_path)

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                   bits_bias=16, bits_sum=32, data_dir=str(tmp_path / "vectors"))
    model.forward(model.quantize_input(X[:1]))

    with pytest.raises(AssertionError, match="B_BITS"):
        check_hardware(model, hw)
