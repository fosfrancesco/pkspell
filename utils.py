from numba import njit
import numpy as np

PAD = "<PAD>"


@njit
def closest_multiple(n: int, x: int):
    if x > n:
        return x
    else:
        return int(x * np.ceil(n / x))


closest_multiple(6900, 2000)
