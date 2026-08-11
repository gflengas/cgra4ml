import os

from deepsocflow.py.brevitas.export import export_inference
from deepsocflow.py.brevitas.hardware import Hardware
from deepsocflow.py.brevitas.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import X, Y

if __name__ == '__main__':
    json_path = os.path.join(os.path.dirname(__file__), 'model', 'xor_graph.json')

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

    # 3. export the golden-reference files a future RTL step will diff against
    #    (batch_size=1 - matches the legacy dense convention, run/param_test.py)
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
        data_dir=os.path.relpath(os.path.join(os.path.dirname(__file__), 'vectors')))

    print()
    result = export_inference(model, hw, X, batch_size=1)
    print(f"Exported golden-reference files: {result['files']}")

    # 4. export the engine-layout files (.txt/.bin, config_fw.h) the RTL
    #    testbench consumes, using all four XOR rows.
    from deepsocflow.py.brevitas.export import export_rtl

    print()
    rtl_result = export_rtl(model, hw, X, batch_size=4)
    print(f"Exported {len(rtl_result['files'])} RTL files to {hw.DATA_DIR}")

    # 5. run the RTL simulation and verify it against the golden reference.
    #    verify_inference reads only the legacy BUNDLES global (already
    #    populated by export_rtl above) and hw - its `model` argument is
    #    unused, so we pass None.
    from deepsocflow.py.xmodel import verify_inference

    # docker_sim.py (used to route hw.simulate() into the pinned-Verilator
    # container, since neither Verilator available on this host can build/run
    # this design - see that script's docstring) monkeypatches
    # deepsocflow.py.hardware.Hardware.simulate - a *different*, non-inheriting
    # class from this backend's own deepsocflow.py.brevitas.hardware.Hardware
    # (kept separate on purpose so this backend doesn't pull in the legacy
    # TensorFlow/qkeras stack - see hardware.py's module docstring). Adopt that
    # patched simulate() only if the legacy module is already imported (i.e.
    # only when actually running under docker_sim.py, which imports it before
    # this script runs) - a plain `python -m deepsocflow.py.brevitas.main`
    # never imports it, so this is a no-op there and TF/qkeras stay unimported.
    import sys
    _legacy_hardware_module = sys.modules.get('deepsocflow.py.hardware')
    if _legacy_hardware_module is not None:
        Hardware.simulate = _legacy_hardware_module.Hardware.simulate

    print()
    hw.export_json()
    hw.export()  # config_hw.svh, config_hw.tcl, sources.txt

    verify_inference(None, hw, SIM='verilator')
    print("brevitas XOR: RTL simulation PASSED")
