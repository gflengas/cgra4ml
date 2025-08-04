# CGRA4ML Investigation for Edge SpAIce

https://github.com/abarajithan11/deepsocflow/assets/26372005/113bfd40-cb4a-4940-83f4-d2ef91b47c91

CGRA4ML is an open-source framework designed to implement large, modern neural networks for scientific edge computing on FPGAs and ASICs. It was created to address the limitations of existing frameworks like HLS4ML, which are highly effective for smaller models but struggle with the size and complexity of modern deep neural networks such as ResNet CNNs, Autoencoders, and Transformers.

This project presents a highly flexible, high performance accelerator system that can be adjusted to your needs through a simple Python API. The implementation is maintained as open source and bare-bones, allowing the user to modify the processing element to do floating point, binarized calculations...etc


## CGRA4ML Overview of a Hybrid System
![System](docs/overall.png)

CGRA4ML provides a complete workflow from model definition to hardware deployment. Its core capabilities include:

*   **Coarse-Grained Reconfigurable Array (CGRA):** Instead of creating a unique spatial datapath for each neural network layer, CGRA4ML generates a reconfigurable array of processing elements (PEs). This allows for resource sharing across different layers, making it highly efficient for large models. The PEs have a simple design, focusing on Computational density and Simplicity.
*   **Support for Large Models:** The framework is explicitly designed to handle modern neural network architectures like ResNet, PointNet, and Transformers, which are often too large for HLS4ML to implement due to on-chip memory constraints.
*   **Off-Chip Data Storage:** A key feature is its ability to manage off-chip data and weight storage, overcoming the primary limitation of HLS4ML, which requires all model parameters to fit in on-chip memory.
*   **Hardware/Software Co-design:** It partitions the neural network, accelerating compute-heavy operations (like convolutions and matrix multiplications) on the CGRA, while executing complex but lightweight pixel-wise operations on a host CPU. The partitions are called "bundles".
*   **Vendor-Agnostic RTL Generation:** CGRA4ML generates SystemVerilog RTL (Register-Transfer Level) code, making it suitable for any ASIC or FPGA design flow, unlike HLS4ML which produces vendor-specific High-Level Synthesis (HLS) code.
*   **End-to-End Toolflow:** The framework includes a Python API built on QKeras for model training, automated generation of C firmware for runtime control, and TCL scripts for both FPGA and ASIC implementation flows. Additionally, it provides a CI/CD integration.
*   **Advanced Verification:** It features a comprehensive verification suite that uses a randomized, transactional SystemVerilog testbench to verify the hardware RTL, the C firmware, and the Python model together, simulating realistic system conditions like memory congestion.

### CGRA4ML Pros and Cons:

| Feature                | Pros                                                                                                                               | Cons                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model Size**         | ✅ **Supports Very Large Models:** Natively designed to handle massive models (e.g., ResNet-50) by using off-chip memory for weights and activations.           | ❌ **Struggles with Small Models:** Can have higher latency and resource overhead for small models where HLS4ML's simpler architecture would be more efficient. |
| **Hardware Target**    | ✅ **ASIC & Vendor-Agnostic Flow:** Generates standard SystemVerilog RTL, providing a clear path from FPGA prototype to a production ASIC.                       | ❌ **Complex Deployment:** Generates a full bare-metal system, which is less flexible and requires significant effort to integrate into a Linux/PetaLinux OS.  |
| **Resource Utilization** | ✅ **Efficient Resource Re-use:** The reconfigurable array (CGRA) is used by all layers, leading to high hardware utilization for deep networks.                  | ❌ **Less Optimized for Low Latency:** The reconfigurable dataflow and off-chip access can introduce latency, making it slower than HLS4ML for certain tasks.   |
| **Architecture**       | ✅ **Flexible HW/SW Co-Design:** Allows for a clean split where the CGRA handles heavy compute and a CPU handles complex control flow and unsupported operations. | ❌ **Incomplete Layer Support:** Lacks built-in support for key layers like `UpSampling2D` or `Conv2DTranspose`, making it unable to run a UNET out-of-the-box. |
| **Framework Maturity** | ✅ **Advanced Verification Suite:** Includes a robust SystemVerilog testbench that co-verifies the RTL, firmware, and software model under realistic conditions. | ❌ **Less Mature & Smaller Community:** As a newer framework, it has a smaller user base, less documentation, and is not as battle-tested as HLS4ML.      |
| **Model Format**       | ✅ **Built for Quantization:** Built on QKeras, making quantization a first-class citizen in the workflow.                                                    | ❌ **No ONNX Support:** Lacks a parser for the industry-standard ONNX format, restricting users to the TensorFlow/Keras ecosystem.                      |
---

### Summary: HLS4ML vs. CGRA4ML

*   **Choose `HLS4ML` if:**
    *   Your primary goal is **ultra-low latency**.
    *   Your model is **small enough** to fit entirely within the FPGA's on-chip memory.
    *   You need **broad support for layers (like UNETs)** and model formats (**ONNX**).
    *   You need to easily integrate the accelerator as a **single IP core** into a larger design (e.g., with PetaLinux).

*   **Choose `CGRA4ML` if:**
    *   Your primary challenge is a **very large and deep neural network** that will not fit on-chip.
    *   You need **high throughput** for large batches of data.
    *   Your end goal is an **ASIC**, and you are using an FPGA for prototyping.
    *   You have the engineering resources to **implement missing layers** and integrate the system manually.

## UNET Architecture Support
Based on the currently documented supported layers, CGRA4ML **cannot fully support a standard UNET architecture out-of-the-box**.

The primary reason is the lack of a native up-sampling layer, which is a critical component of the "expansive path" or "decoder" section of a UNET. The purpose of this path is to increase the spatial resolution of the feature maps, and this is typically done using a Transposed Convolution (`Conv2DTranspose`) or an Up-sampling layer (`UpSampling2D`). Neither of these appears in the list of supported layers for CGRA4ML.

However, CGRA4ML **does support many of the other layers** that constitute a UNET:

*   **Encoder Path:** The "contracting path" of a UNET, which consists of convolutions and pooling layers, can be implemented.
    *   `Conv2D` is supported.
    *   `MaxPooling2D` is supported for down-sampling.
    *   Activation functions like `ReLU` are supported.
*   **Skip Connections:** The `Concatenate` layer, used to connect the feature maps from the encoder to the decoder, is also supported.

### Potential Workaround

Given CGRA4ML's hardware/software co-design philosophy, it might be possible to implement a UNET through a hybrid approach:

1.  **CGRA for Heavy Lifting:** The computationally intensive parts of the UNET, such as the convolutions in both the encoder and decoder paths, would be accelerated on the CGRA.
2.  **CPU for Missing Layers:** The up-sampling operations, which are not supported in the hardware, could be offloaded and executed on the host CPU. The data would be passed from the CGRA to the CPU for the up-sampling step and then passed back to the CGRA for the subsequent convolutions.

## HLS4ML Tutorial Ported Into CGRA4ML

Unfortunately, CGRA4ML doesn't offer out of the box support for PYNQ-Z2. The `example.py` can not run directly on the PYNQ-Z2 board to have a first comparison.
It required to reduce the clock to sub-100MHz to have a working design. I managed to have a working design at 100MHz but it required some modifications:

### 1. Updating the pynq_z2.tcl file

The `pynq_z2.tcl` file is used to generate the platform for the PYNQ-Z2 board. 

### 2. Updating the vivado.tcl and axi_cgra4ml files

The `vivado.tcl` file is used to generate the Vivado project for the PYNQ-Z2 board. It needed some big updates in order to be working with both PYNQ-Z2 and ZCU boards. 

Refactored the Vivado TCL script to use **SmartConnect** for the PYNQ-Z2 board, adapting the AXI interconnect and address mapping. Updated the address range and offset for the `axi_cgra4ml_0` IP core. Added logic to copy the HWH file from either .srcs (Vivado 2020.1) or .gen (newer versions).


### 3. Updating the Hardware class 

Added a `board` parameter to the `Hardware.py` class, defaulting to zcu104, and added methods for exporting to JSON and SystemVerilog formats.

### 4. Updating dma_controller

Introduced address pipelining in the DMA controller to fix a timing violation by adding a W_CALC_ADDR state, which calculates the RAM read address before waiting for the RAM. This was done in order to fix the timing violation in the DMA controller and increase the clock frequency up to 100MHz.

Might get reverted if the performance is not stable for the ZCU. 

### 5. Introducing vitis_flow_pynq.tcl

Added a Vitis TCL script to automate the creation of the Vitis project for PYNQ-Z2. This was done in order we can directly work with server-based workflow instead of the GUI suggested by the authors. Need to `ssh -Y` into the server in order to run `xsct vitis_flow_pynq.tcl` and execute the script. 

### 6. Baremetal vs Linux/Notebook 

By default the design is generating a C code to run, by taking advantage of Vitis toolchain. We could also try to run a Python version through the Pynq notebook. 

## Proposal for Upsampling Layer Implementation

Adding `UpSampling2D` is a great alternative to `Conv2DTranspose` for building UNET architectures, and it often leads to a more modular and potentially simpler implementation. Here’s an investigation into how you could add `UpSampling2D` support to the framework.

The core idea is to treat `UpSampling2D` not as a core computation, but as a special data-handling instruction for the hardware. It tells the system to "stretch" the input feature map before feeding it to a subsequent standard convolution.

### 1. Python Frontend: Modifying the `XBundle`

Instead of creating a new, separate `XUpSampling2D` layer, the most seamless approach would be to integrate the up-sampling operation directly into the `XBundle`. This keeps the "one bundle, one hardware transaction" concept intact.

*   **Modify `XBundle.__init__` in `deepsocflow/py/xbundle.py`**:
    *   You would add a new optional parameter, `upsample=None`, to the `XBundle` constructor. This parameter could take a dictionary specifying the up-sampling parameters, like `{'size': (2, 2), 'interpolation': 'nearest'}`.
    *   The hardware implementation would most easily support `'nearest'` interpolation, as it just involves repeating data.

*   **Modify `XBundle.call`**:
    *   Inside the `call` method, if `self.upsample` is not `None`, you would first apply a standard `tf.keras.layers.UpSampling2D` layer to the input tensor before it gets passed to the `self.core` layer (the `XConvBN`).

A user would then define a UNET's expansive path like this, combining the up-sampling and convolution in one bundle definition:

```python
# Fictional example of a UNET block
x = XBundle(
    upsample={'size': (2, 2)},
    core=XConvBN(filters=128, kernel_size=3, ...)
)(x)
```

### 2. Dataflow: Pre-processing the Input Tensor

The dataflow changes would be relatively straightforward. Since the Python `call` method already handles the up-sampling, the main task is to ensure the `export` flow processes the correctly-sized tensor.

*   **Modify `XBundle.export` in `deepsocflow/py/xbundle.py`**:
    *   Before any data reordering happens, you would perform the up-sampling operation on the input tensor (`x_int`). You can do this with a temporary Keras `UpSampling2D` layer.
    *   The resulting *larger* tensor would then be passed to the existing `reorder_x_q2e_conv` function. No new reordering function would be needed for the input, which is a significant simplification compared to the `Conv2DTranspose` approach.

*   **Update Firmware Configuration**:
    *   The Python `export` process would set new flags in the firmware configuration (`config_fw.h`). You'd add parameters to the `Bundle_t` struct, such as `upsample_en`, `upsample_h`, and `upsample_w`, to tell the hardware how to "stretch" the input for that specific bundle.

### 3. Hardware (RTL): Modifying the Input Streamer

The hardware doesn't need a new computational block. It just needs to be told how to stream the input data differently. The changes would be localized to the module that prepares the pixel stream.

*   **Modify `axis_pixels.sv` in `deepsocflow/rtl/`**:
    *   This module is responsible for reading from the AXI DMA stream and preparing pixels for the processing engine.
    *   You would add logic that, when `upsample_en` is active for the current bundle, modifies the streaming behavior.
    *   For `UpSampling2D` of size `(h, w)`, the logic would:
        1.  Read one pixel from the input AXI stream.
        2.  Present that same pixel `w` times to the processing engine. To do this, it would hold the `s_ready` signal to the AXI DMA low to pause the input stream while it sends the repeated pixels.
        3.  After processing a full row of input pixels, it would need to repeat that entire row `h-1` more times. This could be achieved by buffering the row and re-streaming it from the buffer.

This approach effectively makes the processing engine "see" a larger feature map, even though the data stored in DDR is the original, smaller size.

## Proposal of QONNX to QKeras Reconstruction Workflow

### The Core Principle: Manual Reconstruction, Not Automated Conversion

The standard workflow is to design a model in a framework like QKeras and export it to an exchange format like QONNX. The reverse process is not automated and must be done by manually reconstructing the model. The QONNX file serves as the "blueprint" for rebuilding the model in QKeras.

This process can be summarized in four main steps, followed by a crucial verification stage.

### The 4-Step Reconstruction Process

#### 1. Analyze the Blueprint (The QONNX File)
The first step is to thoroughly inspect the QONNX model to understand its structure and parameters.

*   **Tool:** Use **Netron**, a visualizer for neural network models.
*   **What to look for:**
    *   **Model Architecture:** Identify the sequence of layers. For a U-Net, pay special attention to the skip connections, which are implemented with `Concat` nodes.
    *   **Quantization Parameters:** For every `Quant` node, meticulously record its attributes: `bit_width`, `scale`, `zero_point`, and `signed`. These define the quantization for weights and activations.
    *   **Weight Tensor Names:** Note the names of the weight and bias inputs for each convolutional or dense layer.

#### 2. Build the Model Skeleton in QKeras
Using the architecture identified in Step 1, write the Python code to define the model's structure.

*   **Framework:** Use the **Keras Functional API**, which is necessary for models with non-sequential connections like the U-Net's skip connections.
*   **Action:** Map the ONNX operators to their QKeras equivalents. Recreate the encoder, bottleneck, and decoder paths, ensuring the `Concatenate` layers correctly implement the skip connections.

#### 3. Configure the Quantizers
This is where you apply the "Q" to your Keras model, using the parameters gathered in Step 1.

*   **Action:** For each QKeras layer (like `QConv2D`), use its `kernel_quantizer`, `bias_quantizer`, and `activation_quantizer` arguments.
*   **Configuration:** Define quantizer functions (e.g., `quantized_bits`, `quantized_relu`) using the `bit_width` and other parameters you recorded from the QONNX file's `Quant` nodes.

#### 4. Load the Pre-trained Weights
The final step is to populate your newly built QKeras model structure with the trained weights from the original model.

*   **Tools:** Use the `onnx` Python library to load the QONNX file and access its weights.
*   **Action:**
    1.  Extract all weight and bias tensors into a Python dictionary, using their names as keys.
    2.  Use the `layer.set_weights()` method in your QKeras model to load the corresponding weights and biases into each layer.

### The Final Goal: Verification

After reconstruction, you must verify that your new QKeras model is a faithful replica of the original.

*   **Method:** Provide the same input tensor to both the original ONNX model and your new QKeras model.
*   **Acceptable Difference:**
    *   **Ideally:** The output tensors should be numerically identical or have a difference close to zero.
    *   **Practically:** The ultimate test is the functional output. For a U-Net, this means the final segmentation masks should be identical or have an Intersection over Union (IoU) greater than 0.99. Any minor numerical errors should not impact the model's final decision.

## Project Structure and Workflow

The `cgra4ml` project is organized into several directories, each with a specific purpose. The overall workflow involves defining a model in Python, exporting it for hardware, and then using the generated RTL, C, and Tcl files to target either an FPGA or an ASIC.

### `run/` Directory

This directory contains Python scripts for defining, training, and verifying neural network models using the `deepsocflow` framework. Each script typically defines a neural network model, specifies hardware parameters, and then runs verification and performance estimation.

- **`example.py`**: A comprehensive example demonstrating the use of `deepsocflow`. It loads the MNIST dataset, defines a `UserModel` with both convolutional and dense layers, trains the model, saves and reloads it, specifies hardware, and then verifies the model's inference and performance.
- Other scripts in this directory (`stuck.py`, `dddd_model.py`, `jettagger.py`, `param_test.py`, `part.py`, `pointnet.py`, `resnet18.py`, `resnet50.py`) serve as various examples and test cases for different models and hardware configurations.

### `run/work/` Directory

This directory is a workspace for the `deepsocflow` tools. It contains the configuration files and other artifacts that are generated when running a model through the `export_inference` flow.

- **`config_fw.h`**: A C header file containing the firmware configuration. It defines the number of "bundles" in the model and an array of `Bundle_t` structs with the detailed parameters for each bundle.
- **`config_hw.svh`**: A SystemVerilog header file containing the hardware configuration parameters for the RTL design.
- **`config_hw.tcl`**: A Tcl script containing the hardware configuration parameters for the synthesis and implementation tools.
- **`config_tb.svh`**: A SystemVerilog header file containing parameters for the testbench.
- **`hardware.json`**: A JSON file that stores the hardware configuration parameters.
- **`sources.txt`**: A text file listing the source files for the RTL design.
- **`vivado_flow.tcl`**: A top-level Tcl script for running the Vivado design flow.

### `deepsocflow/` Directory

This directory contains the core source code for the `deepsocflow` project, organized into subdirectories based on the programming language and functionality.

#### `deepsocflow/py/` Directory

This directory contains the core Python source code for the `deepsocflow` framework.

- **`hardware.py`**: Defines the `Hardware` class, which stores the static parameters of the hardware accelerator.
- **`xmodel.py`**, **`xbundle.py`**, **`xlayers.py`**: These files provide the core abstractions for defining, exporting, and verifying neural network models. `XModel` is the base class for user models, `XBundle` represents a group of layers processed together, and `xlayers.py` defines the custom hardware-aware Keras layers.
- **`dataflow.py`**: Manages the dataflow and performance prediction of the models.
- **`utils.py`**: Contains utility functions and classes, including the `XTensor` class for fixed-point quantization.

#### `deepsocflow/c/` Directory

This directory contains the C source code for the `deepsocflow` runtime, which is responsible for controlling the hardware accelerator.

- **`runtime.h`**: The core of the C runtime, defining data structures, constants, and the main `model_run()` function.
- **`deepsocflow_xilinx.h`**: Provides Xilinx-specific implementations for hardware interaction.
- **`xilinx_example.c`**: An example C program for running a model on a Xilinx platform. This is copy-pasted inside the "hellow_world" example in the Xilinx Vitis IDE, in order to run the project.

#### `deepsocflow/rtl/` Directory

This directory contains the Register Transfer Level (RTL) code for the hardware accelerator, written in SystemVerilog and Verilog.

- **`axi_cgra4ml.v`**: The top-level module for the entire accelerator, instantiating the `dnn_engine`, `dma_controller`, and AXI DMA modules.
- **`dnn_engine.v`**: The top-level module for the DNN engine, which connects the `proc_engine`, `axis_pixels`, and `axis_weight_rotator` modules.
- **`proc_engine.sv`**: The core processing engine, a systolic array of Processing Elements (PEs) that performs the multiply-accumulate (MAC) operations.
- **`dma_controller.sv`**: Manages the data transfers between the processing engine and external memory.
- **`axis_pixels.sv`** and **`axis_weight_rotator.sv`**: These modules handle the pixel and weight streams, respectively, performing padding, shifting, and buffering.
- The `ext/` subdirectory contains external or third-party RTL modules for common AXI-related functionalities.

#### `deepsocflow/tcl/` Directory

This directory contains Tcl scripts for the FPGA and ASIC design flows.

- **`fpga/`**: Contains scripts for targeting Xilinx FPGAs using Vivado and Vitis, including board-specific scripts for the ZCU104 and PYNQ-Z2.
- **`asic/`**: Contains scripts for an ASIC design flow, with scripts for synthesis (Synopsys Design Compiler or Cadence Genus) and place-and-route (Cadence Innovus).

#### `deepsocflow/test/` Directory

This directory contains the tests for the project.

- **`py/`**: Python tests for the `deepsocflow/py` code.
- **`sv/`**: SystemVerilog testbenches for verifying the RTL code.
- **`wave/`**: A directory for storing waveform files from simulation.

### `docs/` Directory

This directory contains the project documentation, which is built using Sphinx. The `source/` subdirectory contains the reStructuredText source files for the documentation.

## Quick Start

0. You need XIlinx Vivado for simulation

1. Clone this repo and install deepsocflow
```bash
git clone https://github.com/abarajithan11/deepsocflow
cd deepsocflow
pip install .
```

2. Run the example
```bash
# Edit SIM and SIM_PATH in the file to match your simulator
cd run/work
python ../example.py
```
### Example.py
```python
from deepsocflow import Bundle, Hardware, QModel, QInput

'''
0. Specify Hardware
'''
hw = Hardware (                          # Alternatively: hw = Hardware.from_json('hardware.json')
        processing_elements = (8, 96)  , # (rows, columns) of multiply-add units
        frequency_mhz       = 250      , #  
        bits_input          = 4        , # bit width of input pixels and activations
        bits_weights        = 4        , # bit width of weights
        bits_sum            = 16       , # bit width of accumulator
        bits_bias           = 16       , # bit width of bias
        max_batch_size      = 64       , # 
        max_channels_in     = 2048     , #
        max_kernel_size     = 13       , #
        max_image_size      = 512      , #
        ram_weights_depth   = 20       , #
        ram_edges_depth     = 288      , #
        axi_width           = 64       , #
        target_cpu_int_bits = 32       , #
        valid_prob          = 0.1      , # probability in which AXI-Stream s_valid signal should be toggled in simulation
        ready_prob          = 0.1      , # probability in which AXI-Stream m_ready signal should be toggled in simulation
        data_dir            = 'vectors', # directory to store generated test vectors
     )
hw.export() # Generates: config_hw.svh, config_hw.tcl, config_tb.svh, hardware.json
hw.export_vivado_tcl(board='zcu104')


'''
1. Build Model 
'''
XN = 1
input_shape = (XN,18,18,3) # (XN, XH, XW, CI)

QINT_BITS = 0
kq = f'quantized_bits({hw.K_BITS},{QINT_BITS},False,True,1)'
bq = f'quantized_bits({hw.B_BITS},{QINT_BITS},False,True,1)'
q1 = f'quantized_relu({hw.X_BITS},{QINT_BITS},negative_slope=0)'    
q2 = f'quantized_bits({hw.X_BITS},{QINT_BITS},False,False,1)'       
q3 = f'quantized_bits({hw.X_BITS},{QINT_BITS},False,True,1)'        
q4 = f'quantized_relu({hw.X_BITS},{QINT_BITS},negative_slope=0.125)'

x = x_in = QInput(shape=input_shape[1:], batch_size=XN, hw=hw, int_bits=QINT_BITS, name='input')

x = x_skip1 = Bundle( core= {'type':'conv' , 'filters':8 , 'kernel_size':(11,11), 'strides':(2,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q1}, pool= {'type':'avg', 'size':(3,4), 'strides':(2,3), 'padding':'same', 'act_str':f'quantized_bits({hw.X_BITS},0,False,False,1)'})(x)
x = x_skip2 = Bundle( core= {'type':'conv' , 'filters':8 , 'kernel_size':( 1, 1), 'strides':(1,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q2}, add = {'act_str':f'quantized_bits({hw.X_BITS},0,False,True,1)'})(x, x_skip1)
x =           Bundle( core= {'type':'conv' , 'filters':8 , 'kernel_size':( 7, 7), 'strides':(1,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':False, 'act_str':q3}, add = {'act_str':f'quantized_bits({hw.X_BITS},0,False,True,1)'})(x, x_skip2)
x =           Bundle( core= {'type':'conv' , 'filters':8 , 'kernel_size':( 5, 5), 'strides':(1,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q4}, add = {'act_str':f'quantized_bits({hw.X_BITS},0,False,True,1)'})(x, x_skip1)
x =           Bundle( core= {'type':'conv' , 'filters':24, 'kernel_size':( 3, 3), 'strides':(1,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q1},)(x)
x =           Bundle( core= {'type':'conv' , 'filters':10, 'kernel_size':( 1, 1), 'strides':(1,1), 'padding':'same', 'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q4}, flatten= True)(x)
x =           Bundle( core= {'type':'dense', 'units'  :10,                                                           'kernel_quantizer':kq, 'bias_quantizer':bq, 'use_bias':True , 'act_str':q4}, softmax= True)(x)

model = QModel(inputs=x_in.raw, outputs=x)
model.compile()
model.summary()

'''
2. TRAIN (using qkeras)
'''
# model.fit(...)


'''
3. EXPORT FOR INFERENCE
'''
SIM, SIM_PATH = 'xsim', "F:/Xilinx/Vivado/2022.1/bin/" # For Xilinx Vivado
# SIM, SIM_PATH = 'verilator', "" # For Verilator

model.export_inference(x=model.random_input, hw=hw)  # Runs forward pass in float & int, compares them. Generates: config_fw.h (C firmware), weights.bin, expected.bin
model.verify_inference(SIM=SIM, SIM_PATH=SIM_PATH)   # Runs SystemVerilog testbench with the model & weights, randomizing handshakes, testing with actual C firmware in simulation

'''
4. IMPLEMENTATION

a. FPGA: Open vivado, source vivado_flow.tcl
b. ASIC: Set PDK paths, run syn.tcl & pnr.tcl
c. Compile C firmware with generated header (config_fw.h) and run on device
'''
```
3. FPGA implementation:

3.1. Generate Bitstream from Vivado:
```bash
# Make sure correct fpga board was specified in the above script. Default is ZCU102
# Open Xilinx Vivado, cd into deepsocflow, and type the following in TCL console
cd run/work
source vivado_flow.tcl
```

3.2. Run on a ZYNQ FPGA:
### Execution API
```c
#define NDEBUG
#include "platform.h"
#include "deepsocflow_xilinx.h"

int main() {

  hardware_setup();
  xil_printf("Welcome to DeepSoCFlow!\n Store weights, biases & inputs at: %p; \n", &mem.w);

  model_setup();
  model_run();    // run model and measure time

  // Print: outputs & measured time
  Xil_DCacheFlushRange((INTPTR)&mem.y, sizeof(mem.y));  // force transfer to DDR, starting addr & length
  for (int i=0; i<O_WORDS; i++)
    printf("y[%d]: %f \n", i, (float)mem.y[i]);
  printf("Done inference! time taken: %.5f ms \n", 1000.0*(float)(time_end-time_start)/COUNTS_PER_SECOND);

  hardware_cleanup();
  return 0;
}
```
- Open Xilinx Vitis
- Create an application project, using `.xsa` generated by running the `run/work/vivado_flow.tcl`
- Right click on application project -> Properties
  - ARM v8 gcc compiler -> Directories -> Add Include Paths: Add absolute paths of `run/work` and `deepsocflow/c`
  - ARM v8 gcc compiler -> Optimization -> Optimization most (-O3)
  - ARM v8 gcc linker -> Libraries -> Add Library: `m` (math library)
- Build, Connect board & launch debug
- Add a breakpoint at `model_setup()`. When breakpoint hits, load `run/work/vectors/wbx.bin` to the address printed.
- Continue - This will run the model and print outputs & execution time

## Results

![Results](docs/results-2.png)

### Results for 8 bit

The dataflow and its implementation results in 5.8× more Gops/mm2, 1.6× more Gops/W, higher MAC utilization & fewer DRAM accesses than the state-of-the-art (TCAS-1, TCOMP), processing AlexNet, VGG16 & ResNet50 at 336.6, 17.5 & 64.2 fps, when synthesized as a 7mm^2 chip usign TSMC 65nm GP.

![Results](docs/results.png)

Performance Efficiency (PE utilization across space & time) and number of DRAM accesses:

![Results](docs/perf.png)
![Results](docs/memory.png) 

