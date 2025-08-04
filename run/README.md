# Run Directory

This directory contains Python scripts for defining, training, and verifying neural network models using the `deepsocflow` framework. Each script typically defines a neural network model, specifies hardware parameters, and then runs verification and performance estimation.

## Files

- **`stuck.py`**: This script defines a `UserModel` with several `XBundle` layers. It seems to be for testing a specific model configuration that may have been causing issues. It saves the model, reloads it, and then uses `pytest` to run a parameterized test to configure hardware, verify inference, and predict performance.

- **`dddd_model.py`**: This script specifies hardware parameters, builds a simple `QModel` with dense layers, exports it for inference, and verifies it with a SystemVerilog testbench.

- **`example.py`**: A comprehensive example demonstrating the use of `deepsocflow`. It loads the MNIST dataset, defines a `UserModel` with both convolutional and dense layers, trains the model, saves and reloads it, specifies hardware, and then verifies the model's inference and performance.

- **`jettagger.py`**: This script defines a `UserModel` with dense layers, likely for a jet tagging application in physics. The data loading part is commented out. It saves an untrained model and then uses `pytest` to test the hardware configuration, inference, and performance.

- **`param_test.py`**: Similar to `example.py`, this script uses the MNIST dataset and defines a `UserModel`. However, with no training epochs, this script focuses on testing different hardware parameterizations using `pytest`.

- **`part.py`**: A partially implemented script. It specifies hardware and starts to define a `QModel` with dense layers, but some model definition parts are commented out. It then exports and verifies this incomplete model.

- **`pointnet.py`**: This script defines a PointNet-like model architecture using `XConvBN` and `XDense` layers. The dataset loading and training sections are commented out. It saves the untrained model and uses `pytest` for hardware configuration, inference verification, and performance prediction.

- **`resnet18.py`**: This script defines a ResNet-18 model using `deepsocflow`'s `XBundle` blocks. The dataset loading and model training are commented out. The script saves the untrained model and then uses `pytest` to run a parameterized test for hardware configuration, inference verification, and performance prediction.

- **`resnet50.py`**: Similar to `resnet18.py`, this script defines a ResNet-50 model. The dataset loading and training are commented out. It saves the untrained model and uses `pytest` for hardware configuration, inference verification, and performance prediction. 