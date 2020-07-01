from pathlib import Path
from types import SimpleNamespace
import uuid
import json

import torch
from sacred import Experiment

import src.ml_helpers as mlh
from src import util
from src import assertions
from src.data_handler import get_data
from src.models.model_handler import get_model


from src.models import updates
ex = Experiment()

torch.set_printoptions(sci_mode=False)
DUAL_LIST = ['iwae_dreg', 'tvo_reparam']

@ex.config
def my_config():
    """
    This specifies all the parameters for the experiment.
    Only native python objects can appear here (lists, string, dicts, are okay,
    numpy arrays and tensors are not). Everything defined here becomes
    a hyperparameter in the args object, as well as a column in omniboard.

    More complex objects are defined and manuipulated in the init() function
    and attached to the args object.

    The ProbModelBaseClass object is stateful and contains self.args,
    so hyperparameters are accessable to the model via self.args.hyper_param
    """
    # learning task
    learning_task = 'continuous_vae'
    model_dir = './models'
    data_dir = './data'
    artifact_dir = './artifacts'

    # Model
    loss = 'elbo'
    hidden_dim = 200  # Hidden dimension of middle NN layers in vae
    latent_dim = 50  # Dimension of latent variable z
    integration = 'left'
    cuda = True
    num_stochastic_layers = 1
    num_deterministic_layers = 2
    learn_prior = False
    activation = None # override Continuous VAE layers

    # Hyper
    K = 5
    S = 10
    lr = 0.001
    log_beta_min = -1.60 # base 10 for log_uniform first beta_1

    # Scheduling
    schedule = 'log'
    partition = None
    knots = None # used in coarse_grain updates = J from App. D or to specify moment targets
    per_sample = False # only fast for moments
    per_batch = False # schedule update per batch

    # Recording
    record = False
    record_frequency = 1
    eval_partition = None  # unused.  possibility to std-ize partitions for evaluation
    verbose = False
    dataset = 'mnist'
    train_data_loader = None
    test_data_loader = None

    schedule_update_frequency = 1  # if 0, initalize once and never update

    # Training
    seed = 1
    epochs = 1000
    batch_size = 1000
    valid_S = 40
    test_S = 5000
    test_batch_size = 1

    optimizer = "adam"
    grad_frequency = 20
    checkpoint_frequency = int(epochs / 5)
    checkpoint = False
    checkpoint = checkpoint if checkpoint_frequency > 0 else False

    test_time = False
    test_frequency = 20
    test_during_training = True
    test_during_training = test_during_training if test_frequency > 0 else False
    train_only = False

    save_grads = False

def init(config, _run):
    args = SimpleNamespace(**config)
    assertions.validate_hypers(args)
    mlh.seed_all(args.seed)

    args.data_path = assertions.validate_dataset_path(args)

    if args.activation is not None:
        if 'relu' in args.activation:
            args.activation = torch.nn.ReLU()
        elif 'elu' in args.activation:
            args.activation = torch.nn.ELU()
        else:
            args.activation = torch.nn.ReLU()

    args._run = _run
    args.model_dir = args.artifact_dir

    if args.checkpoint or args.record:
        unique_directory = Path(args.model_dir) / str(uuid.uuid4())
        unique_directory.mkdir(parents=True)
        args.unique_directory = unique_directory

        # Save args json for grepability
        with open(args.unique_directory / 'args.json', 'w') as outfile:
            json.dump(dict(config), outfile, indent=4)

    args.loss_name = args.loss

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    args.partition_scheduler = updates.get_partition_scheduler(args)
    args.partition = util.get_partition(args)

    args.per_batch = False if (args.per_batch and args.per_sample) else args.per_batch
    args.data_path = Path(args.data_path)
    return args


@ex.capture
def log_scalar(_run=None, **kwargs):
    assert "step" in kwargs, 'Step must be included in kwargs'
    step = kwargs.pop('step')

    for k, v in kwargs.items():
        _run.log_scalar(k, float(v), step)

    loss_string = " ".join(("{}: {:.4f}".format(*i) for i in kwargs.items()))
    print(f"Epoch: {step} - {loss_string}")


@ex.capture
def save_checkpoint(model, epoch, train_elbo, train_logpx, opt, args, _run=None, _config=None):
    path = args.unique_directory / 'model_epoch_{:04}.pt'.format(epoch)

    print("Saving checkpoint: {}".format(path))

    if args.loss in DUAL_LIST:
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_phi': opt[0].state_dict(),
                    'optimizer_theta': opt[1].state_dict(),
                    'train_elbo': train_elbo,
                    'train_logpx': train_logpx,
                    'config': dict(_config)}, path)
    else:
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': opt[0].state_dict(),
                    'train_elbo': train_elbo,
                    'train_logpx': train_logpx,
                    'config': dict(_config)}, path)

    _run.add_artifact(path)


def train(args):
    # read data
    train_data_loader, test_data_loader = get_data(args)

    # attach data to args
    args.train_data_loader = train_data_loader
    args.test_data_loader = test_data_loader

    # Make models
    model = get_model(train_data_loader, args)

    args.train_data_loader = train_data_loader
    args.test_data_loader = test_data_loader
    # Make optimizer
    if args.loss in DUAL_LIST:
        optimizer_phi = torch.optim.Adam(
            (params for name, params in model.named_parameters() if 'encoder' in name), lr=args.lr)
        optimizer_theta = torch.optim.Adam(
            (params for name, params in model.named_parameters() if 'decoder' in name), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        _record = mlh.is_record_time(epoch, args)

        if args.loss in DUAL_LIST:
            train_logpx, train_elbo = model.train_epoch_dual_objectives(
                train_data_loader, optimizer_phi, optimizer_theta, record = _record)
        else:
            train_logpx, train_elbo = model.train_epoch_single_objective(
                train_data_loader, optimizer, record = _record)

        log_scalar(train_elbo=train_elbo, train_logpx=train_logpx, step=epoch)

        if mlh.is_test_time(epoch, args):
            args.test_time = True
            test_logpx, test_kl = model.evaluate_model_and_inference_network(
                test_data_loader)
            log_scalar(test_logpx=test_logpx, test_kl=test_kl, step=epoch)
            args.test_time = False

        if mlh.is_checkpoint_time(epoch, args):
            opt = [optimizer_phi, optimizer_theta] if args.loss in DUAL_LIST else [optimizer]
            save_checkpoint(model, epoch, train_elbo, train_logpx, opt, args)

        if mlh.is_schedule_update_time(epoch, args):
            args.partition = args.partition_scheduler(model, args)

        # ------ end of training loop ---------

    if args.train_only:
        test_logpx, test_kl = 0, 0

    results = {
        "test_logpx": test_logpx,
        "test_kl": test_kl,
        "train_logpx": train_logpx,
        "train_elbo": train_elbo
    }

    return results, model


@ex.automain
def experiment(_config, _run):
    '''
    Amended to return
    '''

    args = init(_config, _run)
    result, model = train(args)

    if args.record:
        model.record_artifacts(_run)

    return result
