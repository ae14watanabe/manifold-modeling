import numpy as np
from .utils import create_grids,expand_sequence
class Space(object):
    def __init__(self, data, range_=[-np.inf, np.inf]):
        self.data = data
        self.n_dimension = data.shape[1]
        self.range_ = expand_sequence(range_)
        self.grids = create_grids(self.range_)


class LatentSpace(Space):
    pass
class ObservedSpace(Space):
    pass

class BaseManifoldModeling(object):

