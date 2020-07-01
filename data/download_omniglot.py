__author__ = "Xinqiang Ding <xqding@umich.edu>"
import pickle
import logging
import numpy as np
import scipy.io
import urllib.request
from pathlib import Path

OMNIGLOT_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

OMNIGLOT_MNIST_DIR = Path('./omniglot')
OMNIGLOT_MNIST_DIR.mkdir(exist_ok=True, parents=True)

path = OMNIGLOT_MNIST_DIR / "chardata.mat"

print("Downloading chardata.mat ...")
urllib.request.urlretrieve(OMNIGLOT_URL, path)


def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')


def make_omniglot():
    omni_raw = scipy.io.loadmat(path)

    train_image = reshape_data(omni_raw['data'].T.astype('float'))
    test_image = reshape_data(omni_raw['testdata'].T.astype('float'))

    train_label = omni_raw['targetchar'].T.astype('float')
    test_label = omni_raw['testtargetchar'].T.astype('float')

    print("Saving data into a pickle file ...")

    data = {'train_image': train_image,
            'train_label': train_label,
            'test_image': test_image,
            'test_label': test_label}

    with open("omniglot.pkl", 'wb') as file_handle:
        pickle.dump(data, file_handle)


def make_binarized_omniglot():
    n_validation = 1345

    omni_raw = scipy.io.loadmat(path)
    logging.info('Loaded {}'.format(path))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

    shuffle_seed = 123
    permutation = np.random.RandomState(
        seed=shuffle_seed).permutation(train_data.shape[0])
    train_data = train_data[permutation]

    x_train = train_data[:-n_validation]
    x_valid = train_data[-n_validation:]
    x_test = test_data

    print("Saving data into a pickle file ...")

    data = {'x_train': x_train,
            'x_valid': x_valid,
            'x_test': x_test}

    with open("binarized_omniglot.pkl", 'wb') as file_handle:
        pickle.dump(data, file_handle)


make_omniglot()
make_binarized_omniglot()