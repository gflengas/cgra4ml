# run/work Directory

This directory is a workspace for the `deepsocflow` tools. It contains the configuration files and other artifacts that are generated when running a model through the `export_inference` flow.

## Files

- **`config_fw.h`**: A C header file containing the firmware configuration. It defines the number of "bundles" in the model and an array of `Bundle_t` structs with the detailed parameters for each bundle. This file is essential for the C runtime to execute the model correctly.

- **`config_hw.svh`**: A SystemVerilog header file containing the hardware configuration parameters. It uses `` `define`` statements to set parameters for the RTL design, such as the PE array dimensions, data bit widths, and memory sizes.

- **`config_hw.tcl`**: A Tcl script containing the hardware configuration parameters for the synthesis and implementation tools (e.g., Vivado).

- **`config_tb.svh`**: A SystemVerilog header file containing parameters for the testbench, such as the clock period and handshake probabilities for simulating AXI-Stream stalls.

- **`hardware.json`**: A JSON file that stores the hardware configuration parameters. This allows for easy saving and loading of hardware configurations.

- **`hs_err_pid*.log`**: A log file from the Java Virtual Machine, likely generated if there was an error during the execution of a tool.

- **`sources.txt`**: A text file listing the source files for the RTL design, which is used by the simulation and synthesis tools.

- **`vivado_flow.tcl`**: A top-level Tcl script for running the Vivado design flow, which likely sources the other Tcl scripts to create a project, synthesize, implement, and generate a bitstream. 