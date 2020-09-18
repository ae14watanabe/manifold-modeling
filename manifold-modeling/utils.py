import numpy as np
def expand_sequence(sequence, n_dimension):
    array = np.array(sequence)
    if array.ndim == 0:
        return
    elif array.ndim == 1:
        return np.tile(array[None, :], n_dimension, axis=0)
    elif array.ndim == 2:
        return array
    else:
        raise ValueError('invalid sequence={}'.format(sequence))

def create_grids(range):