# deepsocflow/py Directory

This directory contains the core Python source code for the `deepsocflow` framework. It provides the classes and functions for defining, exporting, and verifying neural network models for execution on a custom hardware accelerator.

## Files

- **`dataflow.py`**: This file is responsible for managing the dataflow and performance prediction of the models. It contains functions for:
    - Calculating runtime parameters based on hardware and layer shapes (`get_runtime_params`).
    - Reordering weights, biases, and activations to match the hardware's data layout (`reorder_*`).
    - Predicting the performance of a model in terms of clock cycles, memory usage, and utilization (`predict_*_performance`).

- **`hardware.py`**: This file defines the `Hardware` class, which stores the static parameters of the hardware accelerator (e.g., processing elements, clock frequency, data bit widths, memory sizes). It's responsible for:
    - Exporting hardware configurations to SystemVerilog (`.svh`) and TCL (`.tcl`) files for synthesis and simulation.
    - Providing an interface to run simulations using tools like XSim, Icarus, or Verilator.

- **`utils.py`**: This file contains a collection of utility functions and classes used throughout the framework, including:
    - `XTensor`: A class for handling fixed-point quantized tensors.
    - Integer arithmetic helper functions.
    - A global `BUNDLES` list to track the bundles of layers in a model.

- **`xbundle.py`**: This file defines the `XBundle` class, which is a core abstraction representing a group of layers that are processed together by the hardware. It's responsible for:
    - Composing a "core" layer with optional pooling, addition, and flatten/softmax layers.
    - Performing integer simulation of the bundle's forward pass.
    - Exporting the bundle's configuration and data for hardware execution.

- **`xlayers.py`**: This file defines custom Keras layers (`XActivation`, `XConvBN`, `XDense`, `XAdd`, `XPool`) that are aware of the hardware's integer-based computation. These layers wrap `QKeras` layers and add functionality for integer simulation and validation.

- **`xmodel.py`**: This file orchestrates the process of exporting a trained Keras model for inference. It defines:
    - `XModel`: A base class for user-defined models.
    - `export_inference`: A function that performs integer simulation, buffer allocation, performance prediction, and generates the necessary configuration files and data for the C runtime.
    - `verify_inference`: A function that runs the hardware simulation and compares the results with the expected outputs to verify correctness. 