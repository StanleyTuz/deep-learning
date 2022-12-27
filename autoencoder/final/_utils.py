import os
import torch
import torchvision
from torchvision.transforms import ToTensor
from torchvision.utils.data import TensorDataset, DataLoader
from typing import Tuple
import urllib


def get_data(dataset: str, path: str = './data', batch_size: int = 64) -> Tuple[Tuple[int,...], TensorDataset, DataLoader]:
    """Load a dataset."""
    if dataset == 'MNIST':
        return _get_mnist_digits()
    elif dataset == 'faces':
        return _get_frey_faces()


def _get_mnist_digits(path: str, batch_size: int) -> Tuple[Tuple[int,...], TensorDataset, DataLoader]: 
    image_shape = (1, 28, 28)
    ds = torchvision.datasets.MNIST(path, download=True, transform=ToTensor())
    dl = DataLoader(ds, batch_size=batch_size)
    return image_shape, ds, dl


def _get_frey_faces(path: str, batch_size: int) -> Tuple[Tuple[int,...], TensorDataset, DataLoader]:
    from scipy.io import loadmat
    image_shape = (1, 28, 20)

    url = "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_fname = os.path.basename(url)
    data_path = os.path.join(path, data_fname)

    # If it doesn't exist, download it
    if not os.path.exists(data_path):
        f = urllib.request.urlopen(url)
        with open(data_path, 'wb') as fout:
            fout.write(f.read())
    ff = loadmat(data_fname, squeeze_me=True, struct_as_record=False)
    ff = ff['ff']
    ff = torch.tensor(ff.T.reshape((-1, *image_shape))) # batch, channel, h, w

    # Load into TensorDataset, DataLoader
    ds = TensorDataset(ff)
    dl = DataLoader(ds, batch_size=batch_size)

    return image_shape, ds, dl
    