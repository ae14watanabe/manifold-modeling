from gmmzoo.som import SOM
from sklearn.datasets import load_iris
import torch
import matplotlib.pyplot as plt
def _main():
    iris = load_iris()
    X = iris.data

    n_dim_latent=2
    n_grids = 20
    n_epoch = 10
    init='pca'
    shape_latent_space='unit_hypercube'
    schedule_sigma = {'max': 0.5, 'min': 0.1}

    som = SOM(X=torch.tensor(X),n_dim_latent=n_dim_latent,init=init,
              shape_latent_space=shape_latent_space,n_grids=n_grids,n_epoch=n_epoch,
              schedule_sigma=schedule_sigma)

    som.fit()
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    ax.scatter(som.ls.data[:, 0], som.ls.data[:, 1])
    plt.show()
if __name__ == '__main__':
    _main()