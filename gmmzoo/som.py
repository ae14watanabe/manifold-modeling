import torch
from gmmzoo.core import BaseGMM, ObservedSpace, LatentSpace
from tqdm import tqdm
class SOM(BaseGMM):
    def __init__(self, X: torch.Tensor, n_dim_latent: int, init='pca',
                 shape_latent_space='unit_hypercube', n_grids=10,
                 n_epoch=20, schedule_sigma='auto'):
        self.os = ObservedSpace(data=X, grids=None)
        # Set about init value and grids
        if shape_latent_space not in ['pca', 'random']:
            raise ValueError('invalid shape_latent_space={}'.format(shape_latent_space))
        if init not in ['unit_hypercube', 'adjust_data']:
            raise ValueError('invalid init={}'.format())
        init_z = self._initialize_z(n_dim=n_dim_latent, init=init)
        if shape_latent_space == 'unit_hypercube':
            from sklearn.preprocessing import MinMaxScaler
            mmscalar = MinMaxScaler()
            init_z = torch.tensor(mmscalar.fit_transform(init_z.detach.numpy()))
        elif shape_latent_space == 'adjust_data':
            if init == 'random':
                raise ValueError('Not support init={}, shape_latent_space={}'.format(init,shape_latent_space))
            else:
                pass
        else:
            raise ValueError('invalid shape_latent_space={}'.format(shape_latent_space))

        grids = self._create_grids_like(data=init_z, n_grids=n_grids,
                                        include_min_max=True, equal_step=True)

        self.ls = LatentSpace(data=init_z, grids=grids)

        # Set about scheduling sigma
        self.n_epoch = n_epoch
        if schedule_sigma == 'auto':
        elif isinstance(schedule_sigma, dict):
            self.sigma_max = schedule_sigma['max']
            self.sigma_min = schedule_sigma['min']
            if 'tau' not in schedule_sigma.keys():
                self.tau = n_epoch
            else:
                self.tau = schedule_sigma['tau']
        else:
            raise ValueError('invalid schedule_sigma={}'.format(schedule_sigma))

    def fit(self, verbose=True):
        if verbose:
            bar = tqdm(range(self.n_epoch))
        else:
            bar = range(self.n_epoch)

        z = self.ls.data
        x = self.os.data
        zeta = self.ls.grids
        for epoch in bar:
            # Estimate mapping by nadaraya-watson kernel estimator
            sigma = max(self.sigma_min, self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - (epoch / self.tau)))
            sqdist = torch.cdist(zeta, z) ** 2.0
            h = torch.exp(-0.5 * sqdist / (sigma * sigma))
            g = torch.sum(h, dim=1)[:, None]
            r = h / g
            y = torch.mm(r, x)
            # Estimate latent variables
            dist = torch.cdist(x, y)
            z = zeta[dist.argmin(dim=1)]