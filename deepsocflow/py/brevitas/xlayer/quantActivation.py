from typing import Optional

from torch import nn

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.nn.quant_activation import QuantReLU as _QuantReLU
from brevitas.nn.quant_activation import QuantSigmoid as _QuantSigmoid
from brevitas.nn.quant_activation import QuantTanh as _QuantTanh
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

'''
- ReLU /
- Sigmoid /
- Tanh /
- LeakyReLU /
- SiLU /
- SELU
- GELU /
- Softmax
'''

class QuantReLU(_QuantReLU):
    pass

class QuantSigmoid(_QuantSigmoid):
    pass

class QuantTanh(_QuantTanh):
    pass

class QuantLeakyReLU(QuantNLAL):
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

class QuantSiLU(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.SiLU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

class QuantSELU(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.SELU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

class QuantGELU(QuantNLAL):
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.GELU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

class QuantSoftMax(QuantNLAL):
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
