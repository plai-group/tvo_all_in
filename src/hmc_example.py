import torch
from torch import nn
from losses.energies import InterpolatedEnergy
# Change to inherit from Flow Class:  class HMC(Flow2D): or  class HMC(FlowLayer):

class InterpolatedEnergy:
    ''' Parameters 
        ------------
        log_pi0 : function which takes (z, x) arguments
            + log density of "proposal" or initial AIS density, unnormalized or normalized
        log_p11 : function which takes (z, x) arguments
            + log density "proposal" or initial AIS density, unnormalized or normalized

        beta : float
            mixing parameter between [0,1]

        q : float
            specifies q-path (where q=1 is geometric mixture path)

        kwargs0, kwargs1 : dict
            might be passed to energy functions log_pi0, log_pi1 if ever necessary
    '''
    
    def __init__(self, log_pi0, log_pi1, beta, args, q=1, kwargs0 = {}, kwargs1 = {}):
        self.log_pi0 = log_pi0
        self.log_pi1 = log_pi1
        self.beta = beta
        self.q = q 
        self.args = args

        # just in case needed?  not sure
        self.kwargs0 = kwargs0    
        self.kwargs1 = kwargs1

    def log_prob(self, z, x = None):
        if self.q == 1:
            return (1-self.beta) * self.log_pi0(z, x)  + self.beta * self.log_pi1(z,x)
        else:
            raise NotImplementedError
            return q_mixture(z, x, self.log_pi0, self.log_pi1, self.beta, self.q)

    def grad(self, z, x = None, multi_sample = False):
        # used to store result of torch.autograd.grad
        grad_outputs = torch.ones((z.shape[0],))

        if multi_sample: # HMC with K samples per chain
            init = self.log_pi0(z,x)
            target = self.log_pi1(z,x)
            w = target - init
            interpolated = (1-self.beta) * init + self.beta * target
            grad = torch.autograd.grad(interpolated, z, grad_outputs = grad_outputs)[0]
            # TO DO: requires SNIS reweighting of individual energies 
            raise NotImplementedError()
        else:
            log_prob = self.log_prob(z, x)
            grad = torch.autograd.grad(log_prob, z, grad_outputs = grad_outputs)[0]

        # clip by norm to avoid numerical instability
        grad = torch.clamp(
            grad, -z.shape[0] *
            self.args.latent_dim * 100,
            z.shape[0] * self.args.latent_dim 
            * 100)
        

        grad.requires_grad_()
        return grad, log_prob

    def grad_and_log_prob(self, z, x = None, multi_sample = False):
        return self.grad(z, x, multi_sample)


# Eventually of a FLOW (or Transform) base class (e.g. RealNVP, IAF, etc.) e.g. (https://github.com/bayesiains/nflows/tree/master/nflows)
class HMC(nn.Module):

    '''
    Arguments
    ----------
        energy_z : InterpolatedEnergy  (taking input (z, x = None) )
            "energy function" for which we follow gradients
            *** energy might imply -log πβ, but let's keep this as log πβ unless confusing
        
        Unnecessary to use energy_v (MH is separate operation)
    
        epsilon : float
            step size

        num_leapfrog : int
            number of leapfrog steps (should this be specified elsewhere)

    '''
    def __init__(self, energy_z, #energy_v, \
                       #epsilon, num_leapfrog, \
                       #adaptive_step_size_args = None, \
                       args):

        # way of specifying these should change...
        self.epsilon = args.hmc_epsilon # should be adaptive tensor for each dim of Z 
        self.num_leapfrog = args.num_leapfog

        self.energy_z = energy_z
        super().__init__()

    def forward(self, initial_samples, initial_weights, x, energy_z = None):
        
        '''
        Arguments
        ----------
        initial_samples = (initial_z, initial_v) : tensors
            assume initial point of trajectory is given 
                (TO DO: incorporate sampling here if useful)
                
        initial_weights : tensor.   (or could be handled elsewhere)
        
        energy_z : losses.energies.InterpolatedEnergy
            option to replace (if necessary?)

        Returns
        ----------
        final_samples = (current_z, current_v) : tensors

        Δ log density ( = 0 for hamiltonian dynamics), or initial_weights + Δ log density

        '''
        if energy_z is not None:
            self.energy_z = energy_z

        current_z = initial_samples[0]
        current_v = intital_samples[1]

        # see InterpolatedEnergy.grad
        grad_Uz, _ = self.energy_z.grad(current_z, x)

        v = current_v - 0.5 * grad_Uz * self.epsilon

        for i in range(1, self.num_leapfrog + 1):

            z = z + v * self.epsilon

            #grad_U_temp = get_grad_U_temp(z, batch, t, **kwargs)
            grad_Uz, _ = self.energy_z.grad(current_z, x)

            if i < self.num_leapfrog:
                v = v - grad_Uz * self.epsilon

        v = v - 0.5 * grad_Uz * self.epsilon
        
        # momentum reversal as separate step?
        #v = -v

        # why this?
        # if not hparams.model_name == 'prior_vae':
        #     z.detach_()
        #     v.detach_()
        return (z, v), initial_weights




class MH(nn.Module):
    '''
    Metropolis-Hastings accept-reject step
    --------------------------------------
    '''
    def __init__(self, energy_z, args):
        '''

        Arguments
        ----------

        '''
        super().__init__()





# Change to inherit from Flow Class:  class MomentumResampling(FlowLayer):

class MomentumResampling(nn.Module):
    def __init__(self, args):
        super().__init__()




class Langevin(nn.Module):

    '''
    Arguments
    ----------
    One-Leapfrog HMC (copy from above)

    '''
    def __init__(self, target_z, target_v, \
                       initial_z, initial_v, \
                       epsilon, adaptive_step_size_args = None, \
                       args):
        super().__init__()
