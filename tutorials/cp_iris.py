from gmmzoo.som import SOM
from sklearn.datasets import load_iris
import torch
def _main():
    iris = load_iris()
    X = iris.data
    som = SOM(X=torch.tensor(X),n_dim_latent=2,init='pca')
if __name__ == '__main__':