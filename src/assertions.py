# Assertions
def validate_hypers(args):
    assert args.schedule in ['log', 'linear', 'moments', 'coarse_grain'], f"schedule cannot be {args.schedule}"
    assert args.integration in ['left', 'right', 'trap', 'single'], f"integration cannot be {args.integration}"
    assert args.loss in ['elbo', 'iwae', 'iwae_dreg', 'tvo', 'tvo_reparam'], f"loss cannot be {args.loss} "
    assert args.learning_task in [ 'continuous_vae'], f" learning_task cannot be {args.learning_task}"
    assert args.dataset in ['omniglot', 'mnist'], f" dataset cannot be {args.dataset} "

    # Add an assertion everytime you catch yourself making a silly hyperparameter mistake so it doesn't happen again


def validate_dataset_path(args):
    dataset = args.dataset

    if dataset == 'mnist':
        data_path = args.data_dir + '/mnist.pkl'
    elif dataset == 'omniglot':
        data_path = args.data_dir + '/omniglot.pkl'
    else:
        raise ValueError("Unknown dataset")

    return data_path
