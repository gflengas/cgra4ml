# deepsocflow Directory

This directory contains the core source code for the `deepsocflow` project, organized into subdirectories based on the programming language and functionality.

## Files

- **`__init__.py`**: This file marks the `deepsocflow` directory as a Python package. It imports modules from the `py/` subdirectory, making them available at the top level of the package.

## Subdirectories

- **`c/`**: Contains C code, likely for firmware or low-level hardware interaction.
- **`py/`**: Contains the core Python code for the `deepsocflow` library, including model definition, hardware abstraction, and dataflow management.
- **`rtl/`**: Contains the Register Transfer Level (RTL) code, written in a Hardware Description Language (HDL) like Verilog or VHDL, which describes the digital circuit's behavior.
- **`tcl/`**: Contains Tcl (Tool Command Language) scripts, typically used for automating interactions with EDA (Electronic Design Automation) tools for synthesis, simulation, and implementation of the hardware design.
- **`test/`**: Contains tests for the project. 