import numpy as np
from .utils import create_grids, expand_sequence
import torch
class Space(object):
    def __init__(self, data:torch.Tensor, grids: torch.Tensor):
        self.data = data.clone()
        self.n_dim = data.shape[1]
        self.grids = grids

        # if range_grids == 'auto':
        # else:
        #     self.range_grids = torch.tensor(expand_sequence(range_grids, n_dimension=self.n_dim))
        # self.grids, self.step = create_grids(n_dim=self.n_dim, n_grids=n_grids, range_=self.range_grids,
        #                                      include_min_max=True, equal_step=True, return_step=True)

    def set_grids(self, grids: np.ndarray):
        self.grids = grids

class LatentSpace(Space):
    pass

class ObservedSpace(Space):
    pass

class BaseGMM(object):
    def __init__(self, X: torch.Tensor, n_dim_latent: int, init='auto',
                 n_grids=10, step_equal=True):
        # grids_x = self._create_grids_like(n_dim=X.shape[0], n_grids=n_grids, include_min_max=True, ste)
        grids_x = None
        self.os = ObservedSpace(data=X, grids=grids_x)
        init_z = self._initialize_z(n_dim=n_dim_latent, init=init)
        grids_z = self._set_grids()
        self.ls = LatentSpace(data=init_z, grids=grids_z)

    def _create_grids_like(self, data, n_grids, include_min_max, equal_step)->torch.Tensor:
        range_grids = torch.cat([data.min(dim=0)[:,None],data.max(dim=0)[:,None]], dim=1)
        return create_grids(n_dim=data.shape[1],n_grids=n_grids,range_=range_grids,
                            include_min_max=include_min_max, equal_step=equal_step)
    def _initialize_z(self, n_dim, init)->torch.Tensor:
        if isinstance(init, (torch.Tensor, np.ndarray)):
            if init.ndim == 2 and init.shape[1] == n_dim:
                return torch.tensor(init)
            else:
                raise ValueError('invalid init={}'.format(init))
        elif init == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_dim)
            return torch.tensor(pca.fit_transform(self.os.data.detach().numpy()))
        elif init == 'random':
            raise ValueError('invalid init={}'.format(init))
        else:
            raise ValueError('invalid init={}'.format(init))

    def fit(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass