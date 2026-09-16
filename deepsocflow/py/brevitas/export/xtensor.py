"""Numpy-only port of deepsocflow/py/utils.py::XTensor, for the brevitas
export path (deepsocflow/py/brevitas/export/rtl_export.py). Same fixed-point
bookkeeping and math as the original - the only difference is that ftensor/
itensor hold numpy arrays instead of tf.Tensor, so this file needs no
tensorflow import at all.

deepsocflow/py/utils.py::XTensor stays exactly as it was for the legacy
qkeras backend; this is a separate, independent copy, not a subclass."""
import numpy as np

from deepsocflow.py.numeric import get_int_bits, get_frac_bits


class XTensor:
    '''Wraps a tensor with fixed-point quantization info, keeping both float (ftensor) and integer (itensor) views in sync'''
    def __init__(self, tensor, bits, frac=None, int=None, float_only=False, from_int=False):
        self.bits = bits            # total bitwidth (int bits + frac bits + 1 sign bit)
        self.float_only = float_only  # True if this tensor has no fixed-point (int/frac) representation
        self.from_int = from_int      # True if constructed directly from already-quantized integer values
        self.error = ""
        if not float_only:
            self.frac = get_frac_bits(bits, int) if frac is None else frac  # number of fractional bits
            self.int = get_int_bits(bits, frac) if int is None else int    # number of integer bits

        tensor = np.asarray(tensor, dtype=np.float32) if tensor is not None else tensor

        if from_int:
            self._itensor = tensor
            self.ftensor = tensor / 2**self.frac
        else:
            self._itensor = None
            self.ftensor = tensor

    @property
    def itensor(self):
        '''Integer (quantized) representation of the tensor'''
        if self.float_only:
            raise ValueError("Only float tensor available")

        if self.from_int:
            return self._itensor
        else:
            return self.ftensor * 2**self.frac

    @property
    def valid(self):
        valid = (self.itensor == self.itensor.astype(int)).all()

        if self.float_only:
            self.error = "Float only"
            return False
        elif not valid:
            self.error = f"Wrong quantization:\n bits:{self.bits}\n frac:{self.frac}\n itensor:{self.itensor}"
            return False
        else:
            return True

    def assert_valid(self):
        assert self.valid, self.error

    def add_val_shift(self, other):
        '''
        Add s,t (self, other) while preserving precision
        '''
        s_intb, t_intb = self.bits-self.frac, other.bits-other.frac  # integer bits of self (s) and other (t)

        r_frac = max(self.frac,other.frac)  # result (r) fractional bits: widest of the two operands
        r_intb = max(s_intb,t_intb)         # result integer bits: widest of the two operands
        r_bits = 1 + r_intb + r_frac # +1 to allow overflow

        s_shift = r_frac-self.frac   # left-shift needed to align self to result's frac bits
        t_shift = r_frac-other.frac  # left-shift needed to align other to result's frac bits

        r = (self.itensor * 2**s_shift) + (other.itensor * 2**t_shift)
        r_tensor = XTensor(tensor=r, bits=r_bits, frac=r_frac, from_int=True)
        return r_tensor, (s_shift, t_shift)
