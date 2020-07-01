from __future__ import division
import torch
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
import joblib
from pathlib import Path

persist_dir = Path('./.persistdir')

class AverageMeter(object):
    """
    Computes and stores the average, var, and sample_var
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.M2   = 0

        self.mean = 0
        self.variance = 0
        self.sample_variance = 0

    def update(self, val):
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

        self.variance = self.M2 / self.count if self.count > 2 else 0
        self.sample_variance = self.M2 / (self.count - 1)  if self.count > 2 else 0

    def step(self,val):
        self.update(val)

class MovingAverageMeter(object):
    """Computes the  moving average of a given float."""

    def __init__(self, name, fmt=':f', window=5):
        self.name = "{} (window = {})".format(name, window)
        self.fmt = fmt
        self.N = window
        self.history = []
        self.val = None
        self.reset()

    def reset(self):
        self.val = None
        self.history = []

    def update(self, val):
        self.history.append(val)
        self.previous = self.val
        if self.val is None:
            self.val = val
        else:
            window = self.history[-self.N:]
            self.val = sum(window) / len(window)
            if len(window)  == self.N:
                self.history == window
        return self.val

    @property
    def relative_change(self):
        if None not in [self.val, self.previous]:
            relative_change = (self.previous - self.val) / self.previous
            return relative_change
        else:
            return 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.relative_change)


def get_data_loader(dataset, batch_size, args, shuffle=True):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """

    if args.device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_mean_of_dataset(train_data_loader, args, idx=0):
    """ Compute mean without loading entire dataset into memory """
    meter = AverageMeter()
    for i in train_data_loader:
        if isinstance(i, list):
            meter.update(i[idx])
        else:
            meter.update(i)
    data_mean = meter.mean
    if data_mean.ndim == 2: data_mean = data_mean.mean(0)
    return tensor(data_mean, args)


def split_train_test_by_percentage(dataset, train_percentage=0.8):
    """ split pytorch Dataset object by percentage """
    train_length = int(len(dataset) * train_percentage)
    return torch.utils.data.random_split(dataset, (train_length, len(dataset) - train_length))


def pmap(f, arr, n_jobs=-1, prefer='threads', verbose=10):
    return Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(delayed(f)(i) for i in arr)

def put(value, filename):
    persist_dir.mkdir(exist_ok=True)
    filename = persist_dir / filename
    print("Saving to ", filename)
    joblib.dump(value, filename)

def get(filename):
    filename = persist_dir / filename
    assert filename.exists(), "{} doesn't exist".format(filename)
    print("Saving to ", filename)
    return joblib.load(filename)

def smooth(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values

def tensor(data, args=None, dtype=torch.float):
    device = torch.device('cpu') if args is None else args.device
    if torch.is_tensor(data):
        return data.to(dtype=dtype, device=device)
    else:
        return torch.tensor(np.array(data), device=device, dtype=dtype)


def is_record_time(epoch, args):
    return args.record and (epoch % args.record_frequency==0 and epoch > 0)


def is_test_time(epoch, args):
    if args.train_only:
        return False

    # last epoch
    if epoch == (args.epochs - 1):
        return True

    # test epoch
    if (args.test_during_training and ((epoch % args.test_frequency) == 0)):
        return True

    # Else
    return False


def detect_cuda(args):
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False
    return args

def is_schedule_update_time(epoch, args):
    # No scheduling
    #if not 'tvo_reparam' in args.loss and not args.loss in ['tvo', 'tvo_2nd_order', 'tvo_3rd_order', 'tvo_reparam', 'tvo_dreg', 'tvo_dreg_q_only', 'tvo_reparam_q_only', 'tvo_reparam_q_final', 'tvo_reparam_q_new', 'annealed_rd']:
    #   return False

    # First epoch, initalize
    if epoch == 0:
        return True

    # Update happens at each minibatch
    if args.per_sample is True:
        return False

    # Initalize once and never update
    if args.schedule_update_frequency == 0:
        return False

    # catch checkpoint epoch
    if (epoch % args.schedule_update_frequency) == 0:
        return True

    # Else
    return False

def is_checkpoint_time(epoch, args):
    # No checkpointing
    if args.checkpoint is False:
        return False

    # skip first epoch
    if (epoch == 0):
        return False

    # catch last epoch
    if epoch == (args.epochs - 1):
        return True

    # catch checkpoint epoch
    if (epoch % args.checkpoint_frequency) == 0:
        return True

    # Else
    return False

def is_gradient_time(epoch, args):
    # No checkpointing
    if args.save_grads is False:
        return False

    # catch checkpoint epoch
    if (epoch % args.grad_frequency) == 0:
        return True

    # Else
    return False

def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def make_sparse(sparse_mx, args):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = tensor(np.vstack((sparse_mx.row, sparse_mx.col)), args, torch.long)
    values = tensor(sparse_mx.data, args)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def converged(loss, previous_loss, thresh=1e-3, check_increased=True, raise_on_increased=False, maximize=False):
    eps = np.finfo(float).eps
    inf = float('inf')
    converged = False
    assert not torch.isnan(loss), '****** loss is nan ********'

    diff = previous_loss - loss
    delta_diff = abs(diff)
    avg_diff = (abs(loss) + abs(previous_loss) + eps) / 2

    if check_increased:
        if maximize:
            if diff > 1e-3:  # allow for a little imprecision
                print('******loss decreased from %6.4f to %6.4f!\n' % (previous_loss, loss))
                if raise_on_increased:
                    raise ValueError
        else:
            if diff < -1e-3:  # allow for a little imprecision
                print('******loss increased from %6.4f to %6.4f!\n' % (previous_loss, loss))
                if raise_on_increased:
                    raise ValueError

    if (delta_diff == inf) & (avg_diff == inf):
        return converged

    if (delta_diff / avg_diff) < thresh:
        converged = True

    return converged

def get_grads(model):
    return torch.cat([torch.flatten(p.grad.clone()) for p in model.parameters()]).cpu()

def log_ess(log_weight):
    """Log of Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: log of effective sample size [batch_size] (or [1])
    """
    dim = 1 if log_weight.ndimension() == 2 else 0

    return 2 * torch.logsumexp(log_weight, dim=dim) - \
        torch.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))
