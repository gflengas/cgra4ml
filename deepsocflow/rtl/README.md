# deepsocflow/rtl Directory

This directory contains the Register Transfer Level (RTL) code for the `deepsocflow` hardware accelerator, written in SystemVerilog and Verilog.

## Files

- **`dnn_engine.v`**: The top-level module for the DNN engine. It instantiates and connects the `proc_engine`, `axis_pixels`, and `axis_weight_rotator` modules.

- **`proc_engine.sv`**: The core processing engine, which is a systolic array of Processing Elements (PEs) that performs the multiply-accumulate (MAC) operations.

- **`dma_controller.sv`**: This module acts as the bridge between the processing engine and the external memory. It generates DMA descriptors and manages the data transfers for weights, pixels, and outputs.

- **`axi_cgra4ml.v`**: The top-level module for the entire accelerator. It instantiates the `dnn_engine`, `dma_controller`, and the AXI DMA modules. It exposes the AXI-Lite slave interface for configuration and the AXI master interfaces for DMA.

- **`axis_pixels.sv`**: This module handles the pixel stream from the DMA, performing padding and shifting to create the sliding window required for convolutions.

- **`axis_weight_rotator.sv`**: This module manages the weight stream, using double buffering to ensure a continuous flow of weights to the processing engine.

- **`ram.sv`**: This file defines several RAM modules used in the design, including `ram_weights`, `ram_edges`, and `ram_output`.

- **`cyclic_bram.sv`**: Implements a cyclic BRAM, which is used for creating circular buffers.

- **`counter.sv`**: A generic, parameterized counter module.

- **`n_delay.sv`**: A generic, parameterized delay chain module.

- **`defines.svh`**: A SystemVerilog header file that includes hardware parameters from `config_hw.svh` and defines the `tuser_st` struct for passing metadata in AXI-Stream interfaces.

### `ext/` Directory

This directory contains external or third-party RTL modules, primarily from a library named "alex," which provide common AXI-related functionalities:
- **`alex_axi_dma_wr.sv` / `alex_axi_dma_rd.sv`**: AXI DMA write and read modules.
- **`alex_axilite_*.sv`**: Modules for AXI-Lite interfaces.
- **`alex_axis_*.sv`**: Modules for AXI-Stream adapters and registers.
- **`xilinx_*.sv`/`.v`**: Xilinx-specific primitives like `xilinx_sdp` (Simple Dual-Port RAM) and `xilinx_spwf` (Simple-Port with Write-First). 