import torch
from gmmzoo.core import BaseGMM
from tqdm import tqdm
class SOM(BaseGMM):
    def fit(self, n_epoch, verbose=True):
        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)

        for epoch in bar:
