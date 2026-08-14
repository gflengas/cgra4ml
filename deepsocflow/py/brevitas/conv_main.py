"""Drives one conv bring-up stage (see conv.py) all the way to RTL simulation.

    python deepsocflow/py/brevitas/conv_main.py --stage a --sim xsim

Mirrors main.py's structure for the XOR model. The two Hardware settings that
are load-bearing rather than arbitrary are called out at the constructor.
"""
import argparse
import os

import numpy as np
import torch

BREV_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='a', choices=list('abcdefghi'))
    # The legacy conv path has only ever been exercised at batch 1 (run/example.py),
    # and the XOR model never exercised it at all - export_bundle's dense branch
    # reshapes a batch into the height axis, so XN stays 1 there too. Keep this
    # adjustable while that assumption is being tested.
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--sim', default='xsim', choices=['xsim', 'verilator', 'none'])
    parser.add_argument('--sim-path', default='/home/software/Xilinx/Vivado/2023.2/bin/')
    args = parser.parse_args()

    from deepsocflow.py.brevitas.conv import (
        build_model, stage_data, prime_batchnorm, MODEL_DIR, STAGE_RESIDUALS)
    from deepsocflow.py.brevitas.hardware import Hardware
    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.sim import FixedPointModel

    # Input size is per-stage: the strided stage needs an odd one (see conv.py).
    X, _, x_rtl_full = stage_data(args.stage)
    x_rtl = x_rtl_full[:args.batch]

    graph_json = os.path.join(MODEL_DIR, f'conv_{args.stage}_graph.json')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- quantize -------------------------------------------------------
    # Stages a-d are untrained by design (see conv.py's module docstring):
    # bit-exactness is a property of the export path, not of the weights.
    model = build_model(args.stage)
    # Stages with BatchNorm need real running statistics or the fold degenerates
    # to the identity and tests nothing (see conv.py::prime_batchnorm).
    prime_batchnorm(model, X)

    # bias_bits=16, not Int32Bias's default of 32, so it fits hw.B_BITS below.
    # The bias's frac is derived (input_frac + weight_frac) rather than chosen,
    # so narrowing this too far saturates every bias value silently - 16 leaves
    # real integer headroom at these scales.
    qm = quantized_model(model, weight_bits=8, bias_bits=16,
                         residuals=STAGE_RESIDUALS.get(args.stage))
    qm.quantization(X)
    qm.eval()

    qm.export_graph_json(x_rtl, graph_json)
    print(f"exported graph json to {graph_json}")

    # ---- integer reference ---------------------------------------------
    fp = FixedPointModel(graph_json)
    fp.load_int_weights(graph_json)
    x_int = fp.quantize_input(x_rtl)
    out_int = fp.forward(x_int)
    print(f"int output shape: {out_int.shape}")
    fp.print_graph()

    # ---- compare against brevitas itself --------------------------------
    # The whole point of this check is that sim.py and the exporter can agree
    # with each other while both disagreeing with the model, which is exactly
    # what a wrong NCHW/NHWC permutation produces: the numbers stay
    # self-consistent and only the arrangement is wrong. So compare against
    # brevitas's own output, dequantized, rather than against anything derived
    # from sim.py.
    with torch.no_grad():
        qm_out = qm(x_rtl)
    qm_ref = qm_out.value if hasattr(qm_out, 'value') else qm_out
    qm_ref = qm_ref.detach().numpy()

    last = fp.bundles[fp.bundle_order[-1]]

    if last['softmax']:
        # forward() returns the PRE-softmax logits while brevitas returns
        # probabilities, so the comparison has to be made on the softmax output.
        # That one cannot be exact: sim.py evaluates softmax in float64 and
        # brevitas in float32, over two independently ordered but mathematically
        # equal expressions. The integer part - everything the hardware actually
        # computes - is still exact, and the decision the model makes is checked
        # exactly via argmax.
        sim_deq = np.asarray(fp.softmax_out, dtype=np.float64)
        assert sim_deq.shape == qm_ref.shape, (
            f"shape mismatch vs brevitas: sim {sim_deq.shape} != brevitas {qm_ref.shape}")
        max_err = float(abs(sim_deq - qm_ref).max())
        n_flips = int((sim_deq.argmax(-1) != qm_ref.argmax(-1)).sum())
        print(f"vs brevitas: max abs err {max_err:.3e} (softmax, float64 vs float32), "
              f"{n_flips}/{len(sim_deq)} predictions differ")
        assert n_flips == 0, f"{n_flips} predictions disagree with brevitas"
        assert max_err < 1e-5, (
            f"softmax max abs err {max_err} is larger than float-rounding noise")
    else:
        sim_deq = out_int.astype('float64') / 2 ** last['act_frac']
        if last['type'] == 'conv':
            sim_deq = sim_deq.transpose(0, 3, 1, 2)  # NHWC -> NCHW to match torch

        assert sim_deq.shape == qm_ref.shape, (
            f"shape mismatch vs brevitas: sim {sim_deq.shape} != brevitas {qm_ref.shape}")
        max_err = float(abs(sim_deq - qm_ref).max())
        n_diff = int((sim_deq != qm_ref).sum())
        print(f"vs brevitas: max abs err {max_err:.3e}, {n_diff}/{sim_deq.size} values differ")
        assert max_err == 0.0, (
            f"sim.py disagrees with brevitas (max abs err {max_err}) - the integer model "
            f"must reproduce the quantized model exactly, not approximately")

    # ---- export + simulate ----------------------------------------------
    # Every non-default below is copied from run/example.py, the one known-good
    # conv configuration in this repo - the XOR model that the brevitas backend
    # was brought up on is dense-only, and export_bundle's dense branch collapses
    # a batch into the height axis, so it never exercised XW > 1 or the edge RAM
    # at all. Defaults that are fine for XOR are not fine here:
    #
    #   valid_prob/ready_prob  Hardware defaults these to 0.01/0.1, which randomly
    #                          throttle the AXI-Stream handshakes to model back
    #                          pressure. At 1% valid that is ~100 idle cycles per
    #                          beat - survivable for XOR's 4 pixels, but it turns
    #                          an 8x8 image into an apparently-hung simulation.
    #   ram_edges_depth        holds the row overlap a KH>1 kernel needs between
    #                          blocks; 288 is not enough at this image width.
    #   axi_width=128          at the default of 64 the simulation completes and
    #                          reports success while silently corrupting bundle 0
    #                          (see CLAUDE.md's Known Issues).
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        ram_edges_depth=3584,
        axi_width=128,
        valid_prob=1, ready_prob=1,
        data_dir=os.path.relpath(os.path.join(BREV_DIR, f'vectors_conv_{args.stage}')))

    from deepsocflow.py.brevitas.export import export_rtl
    from deepsocflow.py.brevitas.rtl_export import verify_inference

    result = export_rtl(fp, hw, x_rtl, batch_size=x_rtl.shape[0])
    print(f"exported {len(result['files'])} RTL files to {hw.DATA_DIR}")

    hw.export_json()
    hw.export()

    if args.sim == 'none':
        print("skipping simulation (--sim none)")
        return

    verify_inference(None, hw, SIM=args.sim, SIM_PATH=args.sim_path)
    print(f"conv stage {args.stage}: RTL simulation PASSED")


if __name__ == '__main__':
    main()
