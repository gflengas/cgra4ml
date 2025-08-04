# deepsocflow/tcl/fpga Directory

This directory contains Tcl scripts for an FPGA (Field-Programmable Gate Array) design flow, specifically targeting Xilinx FPGAs and using the Vivado and Vitis tools.

## Files

- **`vivado.tcl`**: This is a general-purpose script for running the Vivado design flow. It likely includes commands for:
    - Creating a Vivado project.
    - Adding source files.
    - Running synthesis and implementation (place-and-route).
    - Generating a bitstream, which is used to program the FPGA.

- **`vitis_flow.tcl`**: This script is for running the Vitis unified software platform flow. Vitis is used to build and deploy accelerated applications on Xilinx platforms, and this script likely automates the process of integrating the hardware design with the software application.

- **`zcu104.tcl` / `pynq_z2.tcl`**: These are board-specific Tcl scripts for the Xilinx ZCU104 and PYNQ-Z2 development boards. They likely contain board-specific constraints, such as pin assignments and clock frequencies, and other settings required to target these specific boards.

- **`debug.tcl`**: This script is likely used for debugging the design in the Vivado hardware manager. It may contain commands for setting up probes, triggering on specific events, and analyzing signals in the design. 