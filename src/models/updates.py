import torch
from functools import partial
import numpy as np
from src.util import calc_exp
from src.util import get_total_log_weight

def get_partition_scheduler(args):
    """
    Args:
        args : arguments from main.py
    Returns:
        callable beta_update function

    * callable has interface f(log_iw, args, **kwargs)
    * returns beta_id, or unchanged args.partition, by default

    Beta_update functions:
        *** MUST be manually specified in this function
        * given via args.partition_type and other method-specific args
        * should handle 0/1 endpoint insertion internally
        * may take args.K - 1 partitions as a result (0 is given)

    """

    schedule = args.schedule
    P = args.K - 1

    if schedule == 'moments':
        """
            Grosse et al "Moment Averaged Path"
            Args: knot or target points as list of moment mixing proportions α : α * ELBO + (1-α) * EUBO
         """
        if args.knots is not None and args.knots:
            targets = args.knots
        else:
            targets = list(np.linspace(0.0, 1.0, num=P + 2, endpoint=True))

        return partial(moment_search, per_sample=args.per_sample, targets=targets)

    elif schedule == 'coarse_grain':
        """
            Grosse et al (Eq. 20) gives "optimal linear binning" to allocate K points between η0 and η1
            = proportional to sqrt( 2 * Symm KL )
            = sqrt [ (η1 - η0)(β1 - β0) ]
            https://papers.nips.cc/paper/4879-annealing-between-distributions-by-averaging-moments

            Same update rule, but with given knot points (linspace or args.knots param)
        """
        if args.knots is not None and not isinstance(args.knots, int):
            knots = args.knots if not isinstance(args.knots, list) else list(
                np.linspace(0.0, 1.0, num=args.knots, endpoint=True))
        else:
            num_knots = 21 if not isinstance(args.knots, int) else args.knots
            num_knots = num_knots+1 if num_knots % 5 != 1 else num_knots
            knots = torch.linspace(0.0, 1.0, num_knots)
            #knots = list(np.linspace(0.0,1.0,num=num_knots, endpoint = True))
        return partial(coarse_grain, knots=knots, per_sample=args.per_sample)
    elif schedule in ['log', 'linear']:
        return beta_id


def beta_id(model, args=None, **kwargs):
    """
    dummy beta update for static / unspecified partition_types
    """
    return args.partition

def moment_search(model, args=None, targets = [0.1, 0.25, 0.5, 0.75],
                                 threshold = 0.05, start = 0.0, stop = 1.0, partitions = None, per_sample = False):
    """
        Args:  model,
            (args will override model.args)
            (targets set by partition_scheduler)


        targets = desired moment mixtures : α * ELBO + (1-α) * EUBO

        threshold : find target within abs(threshold)

        partitions = # partitions will override model.args
        per sample will override model.args

        Grosse et al "Moment Averaged Path"
        https://papers.nips.cc/paper/4879-annealing-between-distributions-by-averaging-moments
    """
    args = model.args if args is None else args

    if not args.per_sample and not args.per_batch:
        log_iw = get_total_log_weight(model, args, args.valid_S)
    else:
        log_iw = model.elbo()

    partitions = model.args.K-1 if partitions is None else partitions

    left = calc_exp(log_iw, start, all_sample_mean= not(args.per_sample))
    right = calc_exp(log_iw, stop, all_sample_mean= not(args.per_sample))
    left = torch.mean(left, axis = 0, keepdims = True) if args.per_batch else left
    right = torch.mean(right, axis = 0, keepdims= True) if args.per_batch else right
    moment_avg = right - left

    beta_result = []
    for t in range(len(targets)):
        if targets[t] == 0.0 or targets[t] == 1.0:
            # zero if targets[t]=0
            beta_result.append(targets[t] * (torch.ones_like(log_iw[:,0]) if args.per_sample else 1) )
        else:
            target = targets[t]
            moment = left + target*moment_avg #for t in targets]

            start = torch.zeros_like(log_iw[:,0]) if args.per_sample else torch.zeros_like(left)
            stop = torch.ones_like(log_iw[:,0]) if args.per_sample else torch.ones_like(left)

            beta_result.append(_moment_binary_search( \
                    moment, log_iw, start = start, stop = stop, \
                        threshold=threshold, per_sample = args.per_sample))

    if args.per_sample: #or args.per_batch:
        beta_result = torch.cat([b.unsqueeze(1) for b in beta_result], axis=1).unsqueeze(1)
        beta_result, _ = torch.sort(beta_result, -1)
    else:
        beta_result = torch.cuda.FloatTensor(beta_result)

    return beta_result

def _moment_binary_search(target, log_iw, start=0, stop= 1, threshold = 0.1, recursion = 0, per_sample = False, min_beta = 0.001): #recursion = 0,

    beta_guess = .5*(stop+start)
    eta_guess = calc_exp(log_iw, beta_guess, all_sample_mean = not per_sample).squeeze()
    target = torch.ones_like(eta_guess)*(target.squeeze())
    start_ = torch.where( eta_guess <  target,  beta_guess, start)
    stop_ = torch.where( eta_guess >  target, beta_guess , stop)

    if torch.sum(  torch.abs( eta_guess - target) > threshold ).item() == 0:
        return beta_guess
    else:
        if recursion > 500:
            return beta_guess
        else:
            return _moment_binary_search(
                target,
                log_iw,
                start= start_,
                stop= stop_,
                recursion = recursion + 1,
                per_sample = per_sample)


def coarse_grain(model, args=None, knots=None, partitions=None, per_sample=False,  **kwargs):
    """

    Args : model  :  partitions, args, per_sample override model.args

    knots = points between which to apply Grosse et al (Eq. 20)

        between any two points, allocate proportional to
                    sqrt [ (η1 - η0)(β1 - β0) ] = sqrt ( 2 * symm kl)
        https://papers.nips.cc/paper/4879-annealing-between-distributions-by-averaging-moments

    note: function rounds to nearest integer and allocates any remainders based on abs( x - round(x) )
    """

    if not args.per_sample:
        log_iw = get_total_log_weight(model, args, args.valid_S)
    else:
        log_iw = model.elbo()

    if args is None:
        args = model.args
    if partitions is None:
        partitions = model.args.K-1
    if knots is None:
        if args.knots is not None and args.knots:
                knots = args.knots
        else:
                #knots = torch.linspace(0.0, 1.0, partitions+2, device='cuda')
                knots = list(np.linspace(0.0, 1.0, 51, endpoint = True))


    # calculate values at each knot point
    costs = []
    for _k in range(1,len(knots)):
            k = knots[_k]
            prev_k = knots[_k-1]

            k_exp = calc_exp(log_iw, k, all_sample_mean = not per_sample)
            prev_exp = calc_exp(log_iw, prev_k, all_sample_mean = not per_sample)


            cost = (k-prev_k)*(k_exp-prev_exp)
            cost = torch.where(cost > 0, cost, torch.zeros_like(cost))
            costs.append(cost)

    costs = torch.stack(costs, axis=-1)

    sqrt_costs = torch.sqrt(costs)
    alloc = partitions*sqrt_costs/torch.sum(10**-10+sqrt_costs, axis =-1).unsqueeze(-1)

    knots = torch.linspace(0.0, 1.0, partitions+2, device='cuda')

    num_to_fix = partitions - torch.sum(torch.round(alloc), axis=-
                           1).unsqueeze(-1)
    sorts, argsorts = torch.sort(
        (alloc - torch.round(alloc)).squeeze(), axis=-1)


    alloc = alloc.squeeze()

    if torch.sum(num_to_fix != 0) > 0:
        for i in range(num_to_fix.shape[0]):
            off = 1
            for j in range(int(torch.abs(num_to_fix[i].squeeze()).item())):
                if (num_to_fix[i].squeeze()).item() < 0:
                    if len(alloc.shape) > 2 and torch.round(alloc[i, ..., argsorts[i, j+off]]) > 0:
                        alloc[i, ..., argsorts[i, j+off]] -= 1
                    elif len(alloc.shape) == 2 and torch.round(alloc[i, argsorts[i, j+off]]) > 0:
                        alloc[i, argsorts[i,j+off]] -= 1
                    elif len(alloc.shape) == 1 and torch.round(alloc[argsorts[j+off]]) > 0:
                        alloc[argsorts[j+off]] -= 1
                    else:
                        off += 1
                        j -= 1
                elif (num_to_fix[i].squeeze()).item() > 0:
                    if len(alloc.shape) > 2:
                        alloc[i, ..., argsorts[i, -(j+off)]] += 1
                    else:
                        try:
                            alloc[i, argsorts[i, -(j+off)]] += 1
                        except:
                            if len(alloc.shape) == 1:
                                alloc[argsorts[-(j+off)]] += 1

    alloc = torch.round(alloc)
    alloc = alloc.squeeze()


    _off = 0

    alloc = alloc.unsqueeze(0) if len(alloc.shape) == 1 else alloc

    new_betas = [[torch.linspace(knots[_k]+(0 if _k > 0 else knots[_k]+(knots[_k+1]-knots[_k])/(alloc[_row, _k].item()+1)),
                                 knots[_k+1]-(knots[_k+1]-knots[_k]) /
                                 (alloc[_row, _k].item()+1),
                                 int(alloc[_row, _k].item()), device='cuda')
                  for _k in range(alloc.shape[-1]) if alloc[_row, _k].item() > 0]
                 for _row in range(alloc.shape[0])]

    if len(new_betas) > 1:
        to_cat = [torch.cat([new_betas[i][j] for j in range(len(new_betas[i]))], axis=-1) for i in range(len(new_betas))]

        new_betas = torch.stack(to_cat, axis=0)
        new_betas = torch.cat([torch.zeros_like(new_betas[..., 0]).unsqueeze(
            -1), new_betas, torch.ones_like(new_betas[..., 0]).unsqueeze(-1)], axis=-1)
        #new_beta = new_beta.T
    else:
        new_betas = torch.cat([t.unsqueeze(0) for t in new_betas[0]], axis=1)
        new_betas = torch.cat([torch.zeros_like(new_betas[..., -1]).unsqueeze(-1),
                               new_betas, torch.ones_like(new_betas[..., -1]).unsqueeze(-1)], axis=-1)

    return new_betas.unsqueeze(1) if len(new_betas.shape) < 3 else new_betas
