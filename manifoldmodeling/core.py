import numpy as np
from .utils import create_grids,expand_sequence
class Space(object):
    def __init__(self, data, n_grids, range_):
        self.data = data.copy()
        self.n_dim = data.shape[1]
        self.range_ = expand_sequence(range_, n_dimension=self.n_dim)
        self.grids, self.step = create_grids(n_dim=self.n_dim, n_grids=n_grids, self.range_,
                                             include_min_max=True, equal_step=True, return_step=True)

    def set_grids(self, grids: np.ndarray):
        self.grids = grids

class LatentSpace(Space):
    pass

class ObservedSpace(Space):
    pass

class BaseManifoldModeling(object):
    def __init__(self, X, n_latent_dim, init='auto'):
        self.os = ObservedSpace(data=X, range_='auto')
        self.ls = LatentSpace(data=self._initialize_latent_variable())

    def _initialize_latent_variable(self, init):

    def fit(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass