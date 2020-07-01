import pickle
import torch.utils.data
from src.ml_helpers import tensor, get_data_loader
from torch.utils.data import Dataset


class StochasticMNIST(Dataset):
    def __init__(self, image):
        super(StochasticMNIST).__init__()
        self.image = image

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        return (torch.bernoulli(self.image[idx, :]), )


def make_continuous_vae_data(args):
    # read data
    with args._run.open_resource(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = data['train_image']
    test_image = data['test_image']

    # See page 6, footnote 2 here: https://arxiv.org/pdf/1509.00519.pdf
    train_image = StochasticMNIST(tensor(train_image, args))
    test_image = StochasticMNIST(tensor(test_image, args))

    train_data_loader = get_data_loader(train_image, args.batch_size, args)
    test_data_loader = get_data_loader(test_image, args.test_batch_size, args)
    return train_data_loader, test_data_loader


def get_data(args):
    if args.learning_task in ['continuous_vae']:
        return make_continuous_vae_data(args)
    else:
        raise ValueError(
            "{} is an invalid learning task".format(args.learning_task))
