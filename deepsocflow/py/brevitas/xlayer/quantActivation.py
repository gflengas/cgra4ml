from typing import Optional

from torch import nn

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.nn.quant_activation import QuantIdentity as _QuantIdentity
from brevitas.nn.quant_activation import QuantReLU as _QuantReLU
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

'''
- ReLU /
- LeakyReLU /
- Softmax
'''

class QuantIdentity(_QuantIdentity):
    # Quantizes a tensor with no nonlinearity applied - use this instead of plain
    # torch.nn.Identity() wherever a "no activation" slot still needs to carry real
    # scale/bit-width metadata (e.g. between two convs in a residual block, before a
    # QuantAvgPool2d/QuantAdaptiveAvgPool2d that requires a QuantTensor input, or any
    # point that needs to round-trip through PTQ calibration and QONNX export as a
    # proper quantized value rather than passing through as an unquantized float).
    #
    # Arguments:
    #   - act_quant (optional): quantizer applied to the output. Default: Int8ActPerTensorFloat (8-bit, signed)
    #   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
    #   - bit_width (int, optional): overrides act_quant's bit width, e.g. 4
    #
    # Input:  x, any shape - Tensor or QuantTensor
    # Output: same shape as input - Tensor, or IntQuantTensor if return_quant_tensor=True
    #
    # Usage:
    #   core.act = QuantIdentity()                                    # 8-bit, no nonlinearity
    #   core.act = QuantIdentity(bit_width=4, return_quant_tensor=True)  # feeds e.g. QuantAdaptiveAvgPool2d
    pass

class QuantReLU(_QuantReLU):
    # Arguments:
    #   - act_quant (optional): quantizer applied to the output. Default: Uint8ActPerTensorFloat (8-bit, unsigned)
    #   - input_quant (optional): quantizer applied to the input. Default: None (unquantized)
    #   - bit_width (int, optional): overrides act_quant's bit width, e.g. 4
    #   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
    #
    # Input:  x, any shape - Tensor or QuantTensor
    # Output: same shape as input - Tensor, or IntQuantTensor if return_quant_tensor=True
    #
    # Usage:
    #   relu = QuantReLU()
    #   y = relu(x)  # 8-bit output (default)
    #
    #   relu4 = QuantReLU(bit_width=4, return_quant_tensor=True)
    #   y = relu4(x)  # 4-bit output, returned as IntQuantTensor
    pass

class QuantLeakyReLU(QuantNLAL):
    # Wraps torch.nn.LeakyReLU with output quantization.
    #
    # Arguments:
    #   - act_quant (optional): quantizer applied to the output. Default: Int8ActPerTensorFloat (8-bit, signed)
    #   - input_quant (optional): quantizer applied to the input. Default: None (unquantized)
    #   - bit_width (int, optional): overrides act_quant's bit width, e.g. 4
    #   - return_quant_tensor (bool, optional): return an IntQuantTensor instead of a plain Tensor. Default: False
    #   - **kwargs: forwarded to torch.nn.LeakyReLU (e.g. negative_slope)
    #
    # Input:  x, any shape - Tensor or QuantTensor
    # Output: same shape as input - Tensor, or IntQuantTensor if return_quant_tensor=True
    #
    # Usage:
    #   act = QuantLeakyReLU(bit_width=4)
    #   y = act(x)
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.LeakyReLU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

class QuantSoftMax(QuantNLAL):
    # Wraps torch.nn.Softmax with output quantization. Arguments/Input/Output: same
    # pattern as QuantLeakyReLU (act_quant default: Int8ActPerTensorFloat, 8-bit signed),
    # except passthrough_act=True - quantization noise means the output may not sum to
    # exactly 1.0 per row.
    #
    # Usage:
    #   softmax = QuantSoftMax()
    #   y = softmax(x)  # y.sum(dim=-1) ~= 1.0
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Softmax,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
