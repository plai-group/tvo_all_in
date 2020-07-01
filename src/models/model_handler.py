from src.models.vaes import ContinuousVAE


def get_model(train_data_loader, args):
    if args.learning_task == 'continuous_vae':
        D = train_data_loader.dataset.image.shape[1]
        model = ContinuousVAE(D, args)
    else:
        raise ValueError("Incorrect learning task: {} not valid".format(args.learning_task))
    if args.device.type == 'cuda':
        model.cuda()

    return model
