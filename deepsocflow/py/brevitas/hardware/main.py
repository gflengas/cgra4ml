import os

from deepsocflow.py.brevitas.hardware.hardware import Hardware
from deepsocflow.py.brevitas.simulation.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import X, Y

if __name__ == '__main__':
    # brevitas package root (parent of hardware/), so model/ and vectors/ land
    # in the same place they did before this file moved into hardware/.
    _BREV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(_BREV_DIR, 'model', 'xor_graph.json')

    # 1. build the model's structure from the already-quantized graph JSON
    #    (shapes, activations, topology - weight/bias arrays not populated yet)
    model = FixedPointModel(json_path)
    print(f"Built model '{json_path}' with {len(model.bundle_order)} bundles")

    # 2. only now pull the quantized int weight/bias values in, from the same JSON
    model.load_int_weights(json_path)
    print("Loaded int weights")

    x_int = model.quantize_input(X)
    logits_int = model.forward(x_int)  # pure int64 arithmetic from here on
    preds = logits_int.argmax(axis=-1)  # softmax is monotonic - argmax unaffected

    print()
    print("x_int:", x_int.tolist())
    print("logits_int:", logits_int.tolist())
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())

    print()
    print("Model graph:")
    model.print_graph()

    # 3. export the engine-layout files (.txt/.bin, config_fw.h) the RTL
    #    testbench consumes, using all four XOR rows. This is the only export
    #    call in this script - export_rtl's underlying _export_bundles already
    #    writes the layout-independent golden-reference files (y_exp.txt,
    #    {ib}_y_nhwc_exp.txt) as part of the same pass (xmodel.py:319-334), so
    #    a separate export_inference call here would just get its own output
    #    to that same DATA_DIR deleted and overwritten by this one - dead work
    #    with a misleading print. export_inference itself is still exported
    #    and tested (deepsocflow/py/brevitas/export/export.py) for callers that only
    #    want the golden reference without the engine-layout files.
    # data_dir must be relative to the CURRENT WORKING DIRECTORY, not just to
    # this file: the legacy xmodel.py writes config_fw.h's DATA_DIR macro as
    # a literal f'"../{hw.DATA_DIR}"' (xmodel.py:268), assuming hw.DATA_DIR is
    # already relative to wherever the export ran from - the C testbench then
    # reads that macro verbatim from one level below (build/). An absolute
    # hw.DATA_DIR breaks that concatenation (".." + "/abs/path" resolves to a
    # nonexistent directory) and the compiled binary segfaults on the first
    # fopen(). os.path.relpath keeps the *files* physically under this
    # directory regardless of which directory the script is invoked from,
    # while still producing a value that composes correctly with the fixed
    # "../" prefix.
    # axi_width=128 (not Hardware's default of 64) is load-bearing: at the
    # default, RTL simulation ran to completion but produced a corrupted
    # bundle-0 y_raw (values duplicated/missing across specific offsets in the
    # raw engine-layout output - a data-alignment symptom, not a rounding
    # difference), even though the identical w_int/x_int correctly reproduce
    # the same shape/values through the legacy qkeras XOR reference
    # (run/xor_qkeras.py), which explicitly sets axi_width=128 and passes
    # cleanly. This model's per-bundle transfer sizes (e.g. 16/24-byte x_bpt,
    # 48/72-byte w_bpt) apparently hit a latent AXI-burst/word-count edge case
    # at the narrower bus width that this XOR-sized model is small enough to
    # trigger. Matching the proven-working reference's bus width avoids it;
    # root-causing the width=64 path itself in the legacy DMA/burst-splitting
    # RTL is out of scope here (see task-8 report).
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        axi_width=128,
        data_dir=os.path.relpath(os.path.join(_BREV_DIR, 'vectors')))

    from deepsocflow.py.brevitas.export.export import export_rtl

    print()
    rtl_result = export_rtl(model, hw, X, batch_size=4)
    print(f"Exported {len(rtl_result['files'])} RTL files to {hw.DATA_DIR}")

    # 4. run the RTL simulation and verify it against the golden reference.
    #    verify_inference reads only the legacy BUNDLES global (already
    #    populated by export_rtl above) and hw - its `model` argument is
    #    unused, so we pass None.
    from deepsocflow.py.brevitas.export.rtl_export import verify_inference

    # hw.simulate() runs as-is here: under plain `python -m
    # deepsocflow.py.brevitas.hardware.main` it's this backend's own
    # Hardware.simulate (deepsocflow/py/brevitas/hardware/hardware.py). Under
    # docker_sim.py - the only way
    # to actually simulate on this host, since neither locally available
    # Verilator can build/run this design (see that script's docstring) -
    # docker_sim.py itself monkeypatches both this class and the legacy
    # deepsocflow.py.hardware.Hardware.simulate before this script runs, so
    # there is nothing for main.py to bridge here.
    print()
    hw.export_json()
    hw.export()  # config_hw.svh, config_hw.tcl, sources.txt

    verify_inference(None, hw, SIM='verilator')
    print("brevitas XOR: RTL simulation PASSED")
