from torch import nn
import src.ml_helpers as mlh
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from src.util import compute_tvo_loss, compute_wake_theta_loss, compute_wake_phi_loss, compute_vimco_loss, \
                     compute_tvo_reparam_loss, _get_multiplier, compute_iwae_loss
from src import util
from src import util


class ProbModelBaseClass(nn.Module):
    def __init__(self, D, args):
        """Base class for probabilistic model.
            - Uses internal state to avoid having to pass data around
            - self.set_internals(), self.log_guide(), self.log_prior(), self.sample_latent(),
              must be overwritten by subclass

        Args:
            D (int): [Size of observation dimension]
            S (int, optional): [Number of samples to be used in MC approx]. Defaults to 25.
        """
        super().__init__()

        # Dimensions
        self.D = D
        self.args = args

        self.hist = defaultdict(list)
        self.hist_eval = defaultdict(list)
        self.record_results = defaultdict(mlh.AverageMeter)

        if self.args.loss in ['elbo', 'iwae', 'iwae_dreg', 'tvo_reparam', 'tvo_reparam_q_iwae']:
            print("Reparam turned: ON")
            self.reparam = True
        else:
            print("Reparam turned: OFF")
            self.reparam = False

        # Internal state
        self.x = None  # Observations
        self.y = None  # Labels
        self.z = None  # Latent samples

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    # For debugging
    def show_state(self):
        for a in self:
            print(a)

    def elbo(self):
        """
        Returns: [N, S]
        """
        return self.log_joint() - self.log_guide()

    def set_internals(self, data, S):
        """
        Implemented by subclass

        This sets the internal state variables so all the functions work

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def check_internals(self):
        """Verify internal state variables have been set.
         - False means not used,
         - None means error
        """
        assert self.x is not None, "self.x not set"
        assert self.y is not None, "self.y not set"
        assert self.z is not None, "self.z not set"

    def log_joint(self):
        """
        log p(x, z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        prior = self.log_prior()
        likelihood = self.log_likelihood()
        if prior.ndim == 1:
            N = self.x.shape[0]
            prior = (1 / N) * prior.unsqueeze(0).repeat(N, 1)
            return likelihood + (1 / self.args.batch_size) * prior
        else:
            return prior + likelihood

    def log_prior(self):
        """
        log p(z) or log p(θ), depending on
        if the prior is over latent parameters
        p(z) or global parameters p(θ)

        Implemented by subclass

        Returns: [N, S] or [S]
            p(z) -> [N, S]
            p(θ) -> [S]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def log_likelihood(self):
        """
        log p(x|z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def log_guide(self):
        """
        log q(z|x) or log q(z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def sample_latent(self, S):
        """
        Implemented by subclass

        Note: S is in the *first* index for sample_latent,
        This is done to match pytorch's broadcasting semantics
        * can be anything. i.e. [S, N, D0, D1, ...]

        Returns: [S, *]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    # ============================
    # ---------- Helpers ----------
    # ============================

    def get_test_log_evidence(self, data, S):
        with torch.no_grad():
            self.set_internals(data, S)
            log_weight = self.elbo()
            log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(S)
            iwae_log_evidence = torch.mean(log_evidence)

        return iwae_log_evidence

    def get_test_elbo(self, data, S):
        with torch.no_grad():
            self.set_internals(data, S)
            log_weight = self.elbo()
            elbo = torch.mean(log_weight)
        return elbo

    def get_log_p_and_kl(self, data, S):
        self.set_internals(data, S)
        log_weight = self.elbo()

        log_p = torch.logsumexp(log_weight, dim=1) - np.log(S)
        elbo = torch.mean(log_weight, dim=1)
        kl = log_p - elbo

        return log_p, kl

    def record_artifacts(self, _run):
        """ -- Record artifacts --
            args: sacred _run object
        """
        artifacts = {}
        for eval in [0,1]:
            hist = self.hist_eval if eval else self.hist
            epoch_idx = hist['epoch']
            for k in hist.keys():
                # temp check for jensen gap or other scheduling tool yielding [# β] != K
                if k != 'epoch':
                    if len(hist[k][0].shape) and len(hist[k])>1 and (hist[k][0].shape[-1] != hist[k][1].shape[-1]):
                        hist[k][0] = np.zeros_like(hist[k][1])

                    try:
                        artifacts[k] = pd.DataFrame.from_dict(dict(zip(epoch_idx, hist[k])), orient='index', \
                        columns = ['beta_'+str(i) for i in range(hist[k][0].shape[-1])] \
                                    if len(hist[k][0].shape)>0 else
                                    [k])
                        artifacts[k].index.name = 'epoch'
                    except:
                        print("Bad Artifact:", k)

                #artifacts.append(artifact)
                #names.append(artifact)

            big_artifact_df = pd.concat(list(artifacts.values()), keys=list(artifacts.keys()),axis=1)
            eval_str = '' if not eval else '_eval'
            # To prevent collisions if jobs are run in parallel
            path = self.args.unique_directory / Path('artifacts'+eval_str+'.csv')
            big_artifact_df.to_csv(path)
            # Fed run as argument to train... could also capture?
            _run.add_artifact(path, name ='record'+eval_str)

    def save_record(self, epoch=None, eval=False):
        if eval:
            hist = self.hist_eval
            betas = (self.args.eval_partition
                     if self.args.eval_partition is not None
                     else torch.linspace(0, 1.0, 101, device='cuda')
                     ).cpu().numpy()
        else:
            hist = self.hist
            betas = self.args.partition.cpu().numpy()

        if epoch is None:  # not fully tested
            if eval:
                # self.args.record_every
                epoch = len(hist['epoch'])*self.args.test_frequency
            else:
                epoch = len(hist['epoch'])*self.args.record_frequency
            hist['epoch'].append(epoch)

        for k, meter in self.record_results.items():
            if isinstance(meter, mlh.AverageMeter):
                hist[k].append(meter.mean.detach().cpu().numpy() if isinstance(
                    meter.mean, torch.Tensor) else meter.mean)
                meter.reset()
            else:
                if isinstance(meter, list):
                    hist[k].append(np.concatenate(meter, axis=0))

        if len(betas.shape) == 3 and betas.shape[0] > 1:
            betas = np.squeeze(np.mean(betas, axis=0))
        hist['beta'].append(betas)

        if self.args.verbose:
            print('betas : ', np.mean(betas, axis=0)
                  if len(betas.shape) > 1 else betas)
            print('tvo exp : ', hist['tvo_exp'][-1])
            print('tvo var : ', hist['tvo_var'][-1])
            #print('tvo third : ', hist['tvo_third'][-1])
            #print('tvo fourth : ', hist['tvo_fourth'][-1])

        # reset recording dictionary (since Hist and Hist_EVAL both use this as storage, resetting meter is not enough)
        self.record_results = defaultdict(mlh.AverageMeter)

    def train_epoch_single_objective(self, data_loader, optimizer, record=False):
        train_logpx = 0
        train_elbo = 0

        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()

            loss, logpx, elbo = self.forward(data)

            loss.backward()
            optimizer.step()

            if record:  # self.args.record:
                self.record_stats()

            train_logpx += logpx.item()
            train_elbo += elbo.item()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)

        if record:
            self.save_record()

        return train_logpx, train_elbo


    def train_epoch_dual_objectives(self, data_loader, optimizer_phi, optimizer_theta, record=False):
        train_logpx = 0
        train_elbo = 0

        for idx, data in enumerate(data_loader):
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()

            if self.args.loss == 'tvo_reparam': # p optimized using tvo
                wake_theta_loss = self.get_tvo_loss(data)
            elif self.args.loss == 'iwae_dreg': # p optimized using IWAE (DReG update is only for q)
                wake_theta_loss = self.get_iwae_loss(data)
            else:
                raise ValueError(
                    "{} is an invalid loss".format(self.args.loss))
            wake_theta_loss.backward()
            optimizer_theta.step()

            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()

            if self.args.loss in ['tvo_reparam']:
                sleep_phi_loss = self.get_tvo_reparam_loss(data)
                sleep_phi_loss.backward()
            elif self.args.loss == 'iwae_dreg':
                sleep_phi_loss = self.get_iwae_dreg_loss(data)
                sleep_phi_loss.backward()
            else:
                raise ValueError(
                    "{} is an invalid loss".format(self.args.loss))
            optimizer_phi.step()


            if record: #self.args.record:
                self.record_stats()

            logpx = self.get_test_log_evidence(data, self.args.valid_S)
            elbo = self.get_test_elbo(data, self.args.valid_S)

            train_logpx += logpx.item()
            train_elbo += elbo.item()

            self.last_training_batch = data

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)

        if record: self.save_record() #self.args.record: self.save_record()
        self.last_training_batch = data

        return train_logpx, train_elbo

    def evaluate_model_and_inference_network(self, data_loader):
        log_p_total = 0
        kl_total = 0
        num_data = 0
        with torch.no_grad():
            for data in iter(data_loader):
                log_p, kl = self.get_log_p_and_kl(data, self.args.test_S)
                log_p_total += torch.sum(log_p).item()
                kl_total += torch.sum(kl).item()

                # RB : Added record_stats with evaluation partition
                self.record_stats(eval=True)
                num_data += data[0].shape[0]

            self.save_record(eval=True)
        return log_p_total / num_data, kl_total / num_data

    def evaluate_model(self, data_loader):
        log_px = 0
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                log_px += self.get_test_log_evidence(data, self.args.test_S)
        return log_px / len(data_loader)

    def record_stats(self, eval=False, extra_record=False):  # , data_loader):
        '''
            Records (across β) : expectation / variance / 3rd / 4th derivatives
                curvature, IWAE β estimator
                intermediate TVO integrals (WIP)
        '''

        '''Possibility of different, standardized partition for evaluation?
            - may run with record_partition specified or overall arg'''

        # Always use validation sample size
        S = self.args.valid_S

        # if self.args.record_partition is not None:
        #    partition = self.args.record_partition
        if eval:  # record_partition:
            partition = self.args.eval_partition \
                if self.args.eval_partition is not None \
                else torch.linspace(0, 1.0, 101, device='cuda')
        else:
            partition = self.args.partition

        log_iw = self.elbo().unsqueeze(-1) if len(self.elbo().shape) < 3 else self.elbo()

        log_iw = log_iw.detach()
        heated_log_weight = log_iw * partition

        snis = mlh.exponentiate_and_normalize(heated_log_weight, dim=1)

        # Leaving open possibility of addl calculations on batch dim (mean = False)
        tvo_expectations = util.calc_exp(
            log_iw, partition, snis=snis, all_sample_mean=True)
        tvo_vars = util.calc_var(
            log_iw, partition, snis=snis, all_sample_mean=True)

        # # Using average meter
        # torch.mean(tvo_expectations, dim=0)
        self.record_results['tvo_exp'].update(tvo_expectations.squeeze().cpu())
        self.record_results['tvo_var'].update(
            tvo_vars.squeeze().cpu())  # torch.mean(tvo_vars, dim=0)
        if extra_record:
            tvo_thirds = util.calc_third(
                log_iw, partition, snis=snis, all_sample_mean=True)
            tvo_fourths = util.calc_fourth(
                log_iw, partition, snis=snis, all_sample_mean=True)

            curvature = tvo_thirds/(torch.pow(1+torch.pow(tvo_vars, 2), 1.5))
            iwae_beta = torch.mean(torch.logsumexp(
                heated_log_weight, dim=1) - np.log(S), axis=0)

            self.record_results['tvo_third'].update(
                tvo_thirds.squeeze().cpu())  # torch.mean(tvo_thirds, dim=0)
            self.record_results['tvo_fourth'].update(
                tvo_fourths.squeeze().cpu())  # torch.mean(tvo_fourths, dim = 0)
            # per sample curvature by beta (gets recorded as mean over batches)
            self.record_results['curvature'].update(curvature.squeeze().cpu())
            # [K] length vector of MC estimators of log Z_β
            self.record_results['iwae_beta'].update(iwae_beta.squeeze().cpu())

            if eval:
                left_riemann = util._get_multiplier(partition, 'left')
                right_riemann = util._get_multiplier(partition, 'right')

                log_px_left = torch.sum(left_riemann * tvo_expectations)
                log_px_right = torch.sum(right_riemann * tvo_expectations)

                # self.record_results['betas']=partition

                # KL_Q = direct calculation via iwae_beta
                kl_q = partition*tvo_expectations-iwae_beta
                # KL_Q = KL_LR = integral of variances
                # beta * tvo_var * dbeta
                kl_lr = torch.cumsum(partition*tvo_vars*right_riemann, dim=0)
                kl_rl = torch.flip(torch.cumsum(torch.flip(
                    (1-partition)*tvo_vars*left_riemann, [0]), dim=0), [0])
                #kl_rl_2 = torch.stack([torch.sum(tvo_vars[i:]*left_riemann[i:]) for i in range(tvo_vars.shape[0])])

                self.record_results['direct_kl_lr'].update(
                    kl_q.squeeze().cpu())
                self.record_results['integral_kl_lr'].update(
                    kl_lr.squeeze().cpu())
                self.record_results['integral_kl_rl'].update(
                    kl_rl.squeeze().cpu())

                # FIND log p(x) by argmin abs(kl_lr - kl_rl) => KL[π_α || q] = KL[π_α || p]
                kl_diffs = torch.abs(kl_lr-kl_rl)
                min_val, min_ind = torch.min(kl_diffs, dim=0)
                log_px_jensen = tvo_expectations[min_ind]
                log_px_beta = partition[min_ind]

                # TVO intermediate UB / LB
                tvo_left = torch.cumsum(tvo_expectations*left_riemann, dim=0)
                tvo_right = torch.cumsum(tvo_expectations*right_riemann, dim=0)

                self.record_results['log_px_via_jensen'].update(
                    log_px_jensen.squeeze().cpu())
                self.record_results['log_px_beta'].update(
                    log_px_beta.squeeze().cpu())
                self.record_results['log_px_left_tvo'].update(
                    log_px_left.squeeze().cpu())
                self.record_results['log_px_right_tvo'].update(
                    log_px_right.squeeze().cpu())
                # all intermediate log Z_β via left/right integration (compare with iwae_beta)
                self.record_results['left_tvo'].update(
                    tvo_left.squeeze().cpu())
                self.record_results['right_tvo'].update(
                    tvo_right.squeeze().cpu())

                self.record_results['log_iw_var'].update(
                    tvo_vars[..., 0].squeeze().cpu())

    # ============================
    # ---------- Losses ----------
    # ============================

    def forward(self, data):
        assert isinstance(data, (tuple, list)
                          ), "Data must be a tuple (X,y) or (X, )"

        if self.args.loss == 'elbo':
            loss = self.get_elbo_loss(data)
        elif self.args.loss == 'iwae':
            loss = self.get_iwae_loss(data)
        elif self.args.loss == 'tvo':
            loss = self.get_tvo_loss(data)
        else:
            raise ValueError("{} is an invalid loss".format(self.args.loss))

        logpx = self.get_test_log_evidence(data, self.args.valid_S)
        test_elbo = self.get_test_elbo(data, self.args.valid_S)

        return loss, logpx, test_elbo

    def get_iwae_loss(self, data, set = True):
        '''
        IWAE loss = log mean p(x,z) / q(z|x)
        '''
        assert self.reparam is True, 'Reparam must be on for iwae loss'
        if set:
            self.set_internals(data, self.args.S)


        log_weight = self.elbo()
        return compute_iwae_loss(log_weight)


    def get_iwae_dreg_loss(self, data):
        assert self.reparam is True, 'Reparam must be on for iwae loss'

        #if self.args.stop_parameter_grad:
        self.enable_stop_grads()

        self.set_internals(data, self.args.S) # stop_parameter_grad = self.args.stop_parameter_grad) #True)

        log_weight = self.elbo()

        normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)

        loss = - \
            torch.mean(torch.sum(torch.pow(normalized_weight,2).detach() * log_weight, 1), 0)
        return loss

    def get_elbo_loss(self, data):
        assert self.reparam is True, 'Reparam must be on for elbo loss'
        self.set_internals(data, self.args.S)

        log_weight = self.elbo()
        train_elbo = torch.mean(log_weight)

        loss = -train_elbo
        return loss


    def get_tvo_loss(self, data):
        assert self.reparam is False or self.args.loss == 'tvo_reparam', 'Reparam must be off for TVO loss'
        self.set_internals(data, self.args.S)


        if self.args.per_sample or self.args.per_batch:
            self.args.partition = self.args.partition_scheduler(self, self.args)

        log_weight = self.elbo()
        log_joint = self.log_joint()
        log_guide = self.log_guide()
        loss = compute_tvo_loss(log_weight, log_joint, log_guide, self.args)

        return loss


    def get_tvo_reparam_loss(self, data):
        assert self.reparam is True, 'Reparam must be ON for TVO Higher Order'

        # used to prevent score function gradients
        self.enable_stop_grads()

        self.set_internals(data, self.args.S)

        if self.args.per_sample or self.args.per_batch:
            self.args.partition = self.args.partition_scheduler(self, self.args)

        log_weight = self.elbo()
        log_joint = self.log_joint()
        log_guide = self.log_guide()

        loss = compute_tvo_reparam_loss(log_weight, log_joint, log_guide, self.args)

        return loss
