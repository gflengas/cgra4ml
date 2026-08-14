"""Generates the Vivado bitstream-flow files (config_hw.tcl/.svh, config_tb.svh,
sources.txt, vivado_flow.tcl) for the brevitas XOR model, for a target board.

Deliberately standalone: loads hardware.py directly by file path instead of
`import deepsocflow...`, so it needs only numpy - not the full torch/brevitas/
TensorFlow stack `deepsocflow/__init__.py` otherwise pulls in. This matters
because export_vivado_tcl()/export() write files containing ABSOLUTE paths
derived from wherever this script runs (Hardware.MODULE_DIR = this repo's own
root, computed at runtime) - those paths are only correct on the machine that
actually has Vivado, so this script is meant to run there directly, on a repo
checkout placed on that machine, not to be run once and copied elsewhere.

Usage (on the Vivado machine, from wherever you want the generated files to
land - a fresh directory is fine, matching this project's `run/work`
convention):
    mkdir -p vivado_run && cd vivado_run
    python /path/to/cgra4ml/deepsocflow/py/brevitas/hardware/export_vivado.py --board zcu104

Then, in that same directory:
    vivado -mode batch -source vivado_flow.tcl
"""
import argparse
import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HARDWARE_PY = os.path.join(_THIS_DIR, "hardware.py")

_spec = importlib.util.spec_from_file_location("_brevitas_hardware", _HARDWARE_PY)
_hardware_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hardware_module)
Hardware = _hardware_module.Hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="zcu104",
                         help="board name - must match a "
                              "deepsocflow/tcl/fpga/<board>.tcl script")
    args = parser.parse_args()

    # Same Hardware config as deepsocflow/py/brevitas/hardware/main.py's RTL-sim run -
    # keep these two in sync if that config ever changes, so the bitstream
    # matches what was already verified in simulation.
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        axi_width=128)

    hw.export()
    print("Wrote config_hw.tcl, config_hw.svh, config_tb.svh, sources.txt")

    hw.export_vivado_tcl(board=args.board)
    print(f"Wrote vivado_flow.tcl (board={args.board})")
    print(f"RTL sources referenced from: {hw.MODULE_DIR}/rtl")
    print(f"Board/vivado scripts referenced from: {hw.MODULE_DIR}/tcl/fpga")
    print()
    print("Next: vivado -mode batch -source vivado_flow.tcl")


if __name__ == "__main__":
    main()
