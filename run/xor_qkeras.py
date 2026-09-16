# run/xor_qkeras.py
"""qkeras XOR-equivalent baseline: same architecture as the brevitas XOR
(deepsocflow/py/brevitas/xor.py), independent weights. Its purpose is to produce
a reference set of RTL-facing files for a model of this shape, and to act as the
regression test for refactors of deepsocflow/py/xmodel.py."""
import os
import sys
sys.path.append("../")

import numpy as np
from tensorflow import keras
from keras.layers import Input
from keras.models import Model, save_model
from qkeras.utils import load_qmodel

from deepsocflow import *

SIM = 'xsim' if os.name == 'nt' else 'verilator'

# Matches the brevitas side: 8-bit activations/weights, 16-bit bias.
sys_bits = SYS_BITS(x=8, k=8, b=16)


@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, *args, **kwargs)

        self.b1 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=3, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)))

        self.b2 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=3, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)))

        self.b3 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=2, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)),
            softmax=True)

    def call(self, x):
        x = self.input_quant_layer(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x


x_in = Input((2,), name="input")
user_model = UserModel(sys_bits=sys_bits, x_int_bits=0)
model = Model(inputs=[x_in], outputs=[user_model(x_in)])

save_model(model, "xor_qkeras.h5")
loaded_model = load_qmodel("xor_qkeras.h5")

hw = Hardware(
    processing_elements=(8, 24),
    frequency_mhz=250,
    bits_input=8,
    bits_weights=8,
    bits_sum=32,
    bits_bias=16,
    max_batch_size=64,
    max_channels_in=512,
    max_kernel_size=9,
    max_image_size=512,
    max_n_bundles=64,
    ram_weights_depth=512,
    ram_edges_depth=3584,
    axi_width=128,
    config_baseaddr="B0000000",
    target_cpu_int_bits=32,
    valid_prob=1,
    ready_prob=1,
    data_dir='vectors_qkeras_xor',
)

hw.export_json()
hw = Hardware.from_json('hardware.json')
hw.export()  # config_hw.svh, config_hw.tcl, sources.txt

# batch_size=4 matches the brevitas side (all four XOR rows). The legacy exporter
# feeds random input, not the XOR truth table - fine here, since this baseline
# exists for file structure and RTL-liveness, not for XOR correctness.
export_inference(loaded_model, hw, batch_size=4)
verify_inference(loaded_model, hw, SIM=SIM)
print("qkeras XOR baseline: RTL simulation PASSED")
