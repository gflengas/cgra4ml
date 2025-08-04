# deepsocflow/c Directory

This directory contains the C source code for the `deepsocflow` runtime, which is responsible for controlling the hardware accelerator and running the neural network models.

## Files

- **`runtime.h`**: This is the core of the C runtime. It defines the data structures, constants, and functions for managing the hardware accelerator. This includes:
    - `Bundle_t`: A struct that holds the configuration for a "bundle" of layers.
    - `Memory_st`: A struct that defines the memory layout for weights, biases, inputs, and outputs.
    - `model_run()`: The main function that executes the model on the hardware. It iterates through the bundles and controls the dataflow and computation.
    - Helper functions for quantization, tiling, and memory access.

- **`deepsocflow_xilinx.h`**: This header file provides Xilinx-specific implementations for hardware interaction. It includes functions for:
    - Hardware setup and cleanup (`hardware_setup()`, `hardware_cleanup()`).
    - Timed model execution (`model_run_timed()`).
    - Cache management.

- **`xilinx_example.c`**: An example C program demonstrating how to use the `deepsocflow` runtime on a Xilinx platform. It shows the basic steps of setting up the hardware, running a model, and printing the output.

- **`sim.c`**: A simple C file that includes `runtime.h` and has an empty `main` function. This is likely used as a starting point or a template for simulations. 