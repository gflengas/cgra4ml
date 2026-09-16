import numpy as np

BUNDLES = []


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
