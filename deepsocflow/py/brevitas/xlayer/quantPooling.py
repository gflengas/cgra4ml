from brevitas.nn.quant_pool import TruncAvgPool2d as _TruncAvgPool2d
from brevitas.nn.quant_pool import TruncAdaptiveAvgPool2d as _TruncAdaptiveAvgPool2d
import torch.nn as nn

class QuantAvgPool2d(_TruncAvgPool2d):
		pass

class QuantAdaptiveAvgPool2d(_TruncAdaptiveAvgPool2d):
		pass

class QuantMaxPool2d(nn.MaxPool2d):
		pass

class QuantAdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
		pass