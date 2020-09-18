import numpy as np
from .utils import create_grids
class Space(object):
    def __init__(self, data, range_=[-np.inf, np.inf]):
        self.data = data
        self.n_dimension = data.shape[1]
        self.range_ = range_
        self.grids = create_grids(range_)


class LatentSpace(Space):
    pass
class ObservedSpace(Space):
    pass

class BaseManifoldModeling(object):

