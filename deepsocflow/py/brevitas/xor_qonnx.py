import json
import math
import os

import torch
import torch.nn as nn
from brevitas.quant import Int32Bias
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.graph.calibrate import calibration_mode

from deepsocflow.py.brevitas.utils import reset_bundles
from deepsocflow.py.brevitas.xbundle.xbundle import XBundle
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantLinear
from deepsocflow.py.brevitas.xlayer.quantActivation import QuantSiLU, QuantIdentity
from deepsocflow.py.brevitas.xor import X, Y, MODEL_DIR, MODEL_PATH

GRAPH_JSON_PATH = os.path.join(MODEL_DIR, "xor_graph.json")


class XOR_QONNX(nn.Module):
    # Same architecture as xor.XOR (hidden_1 -> silu -> hidden_2 -> silu -> out ->
    # softmax), but built directly from xlayer/xbundle instead of going through
    # ptq.quantized_model's automatic torch -> brevitas mapping. Uses power-of-two
    # ("fixed-point") scale quantizers, same as ptq.py, so rescaling is a pure
    # bit-shift in hardware.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_1 = QuantLinear(
            2, 3, bias=True,
            weight_quant=Int8WeightPerTensorFixedPoint,
            input_quant=Int8ActPerTensorFixedPoint,
            bias_quant=Int32Bias)
        hidden_1.act = QuantSiLU(act_quant=Int8ActPerTensorFixedPoint)
        self.b1 = XBundle(core=hidden_1)

        hidden_2 = QuantLinear(
            3, 3, bias=True,
            weight_quant=Int8WeightPerTensorFixedPoint,
            input_quant=Int8ActPerTensorFixedPoint,
            bias_quant=Int32Bias)
        hidden_2.act = QuantSiLU(act_quant=Int8ActPerTensorFixedPoint)
        self.b2 = XBundle(core=hidden_2)

        out = QuantLinear(
            3, 2, bias=True,
            weight_quant=Int8WeightPerTensorFixedPoint,
            input_quant=Int8ActPerTensorFixedPoint,
            bias_quant=Int32Bias)
        out.act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
        self.b3 = XBundle(core=out, softmax=True)

    def forward(self, x):
        reset_bundles()
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x

    # Runs PTQ calibration: forwards calibration_data through the model with
    # calibration_mode enabled so each quant layer's scale is set from real
    # activation statistics instead of brevitas's freshly-initialized default.
    def quantization(self, calibration_data):
        self.eval()
        with torch.no_grad(), calibration_mode(self):
            self(calibration_data)
        return self


# xor.XOR's float param names -> XOR_QONNX's param names (same tensors, different owner).
_NAME_MAP = {
    'hidden_1.weight': 'b1.core.weight',
    'hidden_1.bias':   'b1.core.bias',
    'hidden_2.weight': 'b2.core.weight',
    'hidden_2.bias':   'b2.core.bias',
    'out.weight':      'b3.core.weight',
    'out.bias':        'b3.core.bias',
}


def load(model, path=MODEL_PATH):
    float_state = torch.load(path)
    own_state = model.state_dict()
    with torch.no_grad():
        for float_key, our_key in _NAME_MAP.items():
            own_state[our_key].copy_(float_state[float_key])
    return model


def _frac_bits(scale):
    # scale is power-of-two (2^-frac) by construction - see Int*FixedPoint quantizers.
    return round(-math.log2(scale))


def _act_type_name(act):
    return type(act).__name__.replace("Quant", "").lower()


# Walks the model's bundle chain and serializes everything a fixed-point simulator
# needs per layer: hyperparameters, quantized weight/bias INT levels (not float), and
# bit-width/frac-bits for every tensor role (input, weight, bias, activation output).
def export_graph_json(model, sample_input, path):
    model.eval()
    bundles = [("b1", model.b1), ("b2", model.b2), ("b3", model.b3)]

    layers = {}
    x = sample_input
    prev_name = None
    with torch.no_grad():
        for name, bundle in bundles:
            core = bundle.core

            quant_input = core.input_quant(x)
            quant_weight = core.quant_weight(quant_input)
            quant_bias = core.bias_quant(core.bias, quant_input, quant_weight) if core.bias is not None else None

            w_scale = quant_weight.scale.item()
            w_int = (quant_weight.value / w_scale).round().to(torch.int64).tolist()

            layer = {
                "type": "linear",
                "input": prev_name,
                "in_features": core.in_features,
                "out_features": core.out_features,
                "input_bits": int(core.input_quant.bit_width().item()),
                "input_frac": _frac_bits(quant_input.scale.item()),
                "weight": {
                    "bits": int(quant_weight.bit_width.item()),
                    "frac": _frac_bits(w_scale),
                    "values": w_int,
                },
            }
            if quant_bias is not None:
                b_scale = quant_bias.scale.item()
                layer["bias"] = {
                    "bits": int(quant_bias.bit_width.item()),
                    "frac": _frac_bits(b_scale),
                    "values": (quant_bias.value / b_scale).round().to(torch.int64).tolist(),
                }

            x = bundle(x)  # advance to this bundle's real (post-act/softmax) output

            layer["activation"] = _act_type_name(core.act)
            if core.act.act_quant.is_quant_enabled:
                layer["act_bits"] = int(core.act.act_quant.bit_width().item())
                layer["act_frac"] = _frac_bits(core.act.act_quant.scale().item())
            layer["softmax"] = bundle.softmax is not None

            layers[name] = layer
            prev_name = name

    with open(path, 'w') as f:
        json.dump({"layers": layers}, f, indent=2)
    return path


if __name__ == "__main__":
    model = XOR_QONNX()
    model = load(model, path=MODEL_PATH)
    model.quantization(X)

    model.eval()
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=-1)
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())
    print("output:", out.tolist())

    export_graph_json(model, X, GRAPH_JSON_PATH)
    print(f"exported model graph json to {GRAPH_JSON_PATH}")
