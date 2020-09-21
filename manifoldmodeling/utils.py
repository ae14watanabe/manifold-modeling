import numpy as np


def expand_sequence(sequence, n_dimension):
    array = np.array(sequence)
    if array.ndim == 0:
        return np.full(array, n_dimension)
    elif array.ndim == 1:
        return np.tile(array[None, :], reps=(n_dimension, 1))
    elif array.ndim == 2:
        return array
    else:
        raise ValueError('invalid sequence={}'.format(sequence))


def create_grids(n_dim,
                 n_grids=10,
                 range_=[-1, 1],
                 include_min_max=True,
                 equal_step=True,
                 return_step=False,
                 ):
    # exception error
    # about range_
    if isinstance(range_, (list, tuple)):
        range_dims = expand_sequence(range_, n_dim)
    elif isinstance(range_, np.ndarray):
        if range_.ndim == 1:
            range_dims = expand_sequence(range_, n_dim)
        elif range_.ndim == 2:
            if range_.shape[0] != n_dim:
                raise ValueError('invalid range_={}'.format(range_))
            elif range_.shape[1] != 2:
                raise ValueError('invalid range_={}'.format(range_))
            else:
                range_dims = range_
        else:
            raise ValueError('invalid range_={}'.format(range_))
    else:
        raise ValueError('invalid range_={}'.format(range_))
    length_dims = range_dims[:, 1] - range_dims[:, 0]
    if np.all(length_dims == length_dims[0]):
        is_hypercube = True
    else:
        is_hypercube = False

    # about_n_grids
    if isinstance(n_grids, int):
        if equal_step:
            n_grids_dims = n_grids * (length_dims / length_dims.min()).astype(int)
        else:
            n_grids_dims = np.full(n_dim, n_grids)
    elif isinstance(n_grids, (list, tuple)):
        if len(n_grids) != n_dim:
            raise ValueError('invalid n_grids={}'.format(n_grids))
        else:
            n_grids_dims = n_grids
        if equal_step:
            raise ValueError('If n_grids is list or tuple, equal_resolution must be False')
    elif isinstance(n_grids, np.array):
        if n_grids.ndim == 1:
            if len(n_grids) != n_dim:
                raise ValueError('invalid n_grids={}'.format(n_grids))
            else:
                n_grids_dims = n_grids
            if equal_step:
                raise ValueError('If n_grids is 1d ndarray, equal_resolution must be False')
        else:
            raise ValueError('invalid n_grids={}'.format(n_grids))
    else:
        raise ValueError('invalid n_grids={}'.format(n_grids))

    list_grids = []
    if equal_step:
        # calculate step
        index_shortest_dim = np.argmin(length_dims)
        grid1d_min, step = np.linspace(range_dims[index_shortest_dim, 0],
                                       range_dims[index_shortest_dim, 1],
                                       n_grids_dims[index_shortest_dim],
                                       endpoint=include_min_max,
                                       retstep=True)
        for i, (length, range_) in enumerate(zip(length_dims, range_dims)):
            if i == index_shortest_dim:
                list_grids.append(grid1d_min)
            else:
                start = (step / 2.0) + range_[0]
                n_step = int(length / step)
                grid1d = np.arange(n_step) * step + start
                list_grids.append(grid1d)
    else:
        list_steps = []
        for n_grids, range_ in zip(n_grids_dims, range_dims):
            grid1d, step = np.linspace(range_[0], range_[1], n_grids,
                                       endpoint=include_min_max, retstep=True)
            if include_min_max:
                pass
            else:
                grid1d += step / 2.0
            list_grids.append(grid1d)
            list_steps.append(step)
    list_grids = np.meshgrid(*list_grids, indexing='ij')
    grids = np.stack(list_grids, axis=-1)
    grids = grids.reshape(-1, grids.shape[-1])

    if return_step:
        if equal_step:
            return grids, step
        else:
            return grids, list_steps
    else:
        return grids
