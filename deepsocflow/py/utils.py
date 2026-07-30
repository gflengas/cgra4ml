import tensorflow as tf 
from tensorflow import keras
from qkeras import *
import numpy as np

BUNDLES = []

@keras.saving.register_keras_serializable()
class SYS_BITS:
    '''System-wide bitwidths: x=activations, k=kernel/weights, b=bias'''
    def __init__(self, x, k, b):
        self.x = x  # activation (input/output tensor) bitwidth
        self.k = k  # kernel (weight) bitwidth
        self.b = b  # bias bitwidth
    def get_config(self):
        return {'x': self.x, 'k': self.k, 'b': self.b}

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

        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32) if isinstance(tensor, np.ndarray) else tensor

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
        valid = (self.itensor.numpy() == self.itensor.numpy().astype(int)).all()

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




def shift_round(n,s):
    '''Performs integer division with round-to-nearest-even.
        Eq: np.around(n/2**s).astype(int)
        n: number to shift, s: right-shift amount (bits)'''
    half_b = 1<<(s-1) if s>0 else 0
    return (n + half_b - (s>0)*(~(n>>s)&1) ) >> s


def div_round(n,d):
    '''Performs integer division with round-to-nearest-even for d>0.
        Eq: np.around(n/d).astype(int)
        n: numerator, d: denominator'''
    return (n + (d//2) - (~(d|n//d) &1)) // d


def get_int_bits(bits, frac):
    '''Number of integer bits given total bitwidth (bits) and fractional bits (frac)'''
    return bits-frac-1 # we always use signed integer


def get_frac_bits(bits, int_bits):
    '''Number of fractional bits given total bitwidth (bits) and integer bits (int_bits)'''
    return bits-int_bits-1  # we always use signed integer


def clog2(x):
    '''Ceiling of log2(x): number of bits needed to represent x'''
    return int(np.ceil(np.log2(x)))

