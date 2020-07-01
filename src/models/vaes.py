from src.models.base import ProbModelBaseClass
from src.models.mlp import init_mlp, init_two_prong_mlp
import torch
from torch.distributions import Independent
import torch.nn as nn
from src import util

class VAEBaseClass(ProbModelBaseClass):
    def __init__(self, D, args):
        super(VAEBaseClass, self).__init__(D, args)

        self.num_stochastic_layers = args.num_stochastic_layers
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.num_deterministic_layers = args.num_deterministic_layers

        if isinstance(self.latent_dim, int):
            self.latent_dims = [self.latent_dim for _ in range(self.num_stochastic_layers)]

        elif isinstance(self.latent_dim, list):
            assert len(self.latent_dim) == self.num_stochastic_layers
            self.latent_dims = self.latent_dim

        # Gen model
        self.decoders = None
        self.decoder_to_obs = None

        # Inf network
        self.encoder_to_obs = None
        self.encoders = None

        self.layer_0_params = None

        # internal distribution object set by
        # self.set_internals
        self.inf_network = None
        self.stop_grads = False

    # ======== Model methods ========

    def get_decoder_latent_layer_param(self, layer_idx, previous_latent_layer=None):
        # pylint: disable=not-callable,unsubscriptable-object
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx > 0)

        Returns:
            mu for Normal latent of shape [N, latent_dim]
            sig for Normal latent of shape [N, latent_dim]

            if layer_idx is 0, shape [latent_dim],
            otherwise [N, latent_dim]"""

        if layer_idx == 0:
            return self.layer_0_params
        else:
            return self.decoders[str(layer_idx)](previous_latent_layer)

    def get_decoder_latent_layer(self, layer_idx, previous_latent_layer=None):
        # pylint: disable=not-callable,unsubscriptable-object,no-value-for-parameter
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx > 0)

        Returns: Normal distribution with event_shape [latent_dim]
            if layer_idx is 0, batch_shape is [],
            otherwise [N]"""

        mu, sig = self.get_decoder_latent_layer_param(layer_idx, previous_latent_layer)
        return Independent(torch.distributions.Normal(mu, sig), reinterpreted_batch_ndims=1)

    def get_decoder_latent_dist(self):
        """Returns: distribution for all latent layers:

            dist.sample(sample_shape=[sample_shape]) returns
            (latent_0, ..., latent_N) where each latent_n
            is of shape [sample_shape, latent_dim] and latent_0
            corresponds to the latent furthest away from obs

            if latent_n is of shape [batch_shape, latent_dim]
            dist.log_prob(latent_0, ..., latent_N) returns
            sum_n log_prob(latent_n) which is of shape [batch_shape]"""

        latent_dist = util.ChainDistributionFromSingle(self.get_decoder_latent_layer(layer_idx=0))
        for layer_idx in range(1, self.num_stochastic_layers):
            # be careful about closures
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture/2295372
            latent_dist = util.ChainDistribution(
                latent_dist,
                lambda previous_latent_layer, layer_idx=layer_idx: self.get_decoder_latent_layer(
                    layer_idx=layer_idx, previous_latent_layer=previous_latent_layer))
        return latent_dist

    def get_obs_param(self, latent):
        # pylint: disable=not-callable,unsubscriptable-object
        """
        Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [N, latent_dim]
                with values in {0, 1}

        Returns: logits of Bernoulli likelihood of shape
            [N, batch_size]
        """
        latent_layer = latent[-1]
        return self.decoder_to_obs(latent_layer)

    def get_obs_dist(self, latent):
        """Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [N, latent_dim]
                with values in {0, 1}

        Returns: Bernoulli distribution of batch_shape [N] and
            event_shape [batch_size]
        """
        return Independent(torch.distributions.Bernoulli(logits=self.get_obs_param(latent)), reinterpreted_batch_ndims=1)

    def sample_model_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [N, latent_dim]
                with values in {0, 1}
            obs: tensor of shape [num_samples, batch_size]
        """
        decoder_latent_dist = self.get_decoder_latent_dist()

        if self.reparam:
            latent = decoder_latent_dist.rsample((num_samples, ))
        else:
            latent = decoder_latent_dist.sample((num_samples, ))

        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs

    def sample_model_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples, batch_size]
        """
        return self.sample_model_latent_and_obs(num_samples)[1]


    # ======== Inf Network functions ========

    def get_encoder_latent_layer_param(self, layer_idx, previous_latent_layer=None, obs=None):
        # pylint: disable=not-callable,unsubscriptable-object
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [N, batch_size] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns:
            mu for Normal latent of shape [N, latent_dim]
            sig for Normal latent of shape [N, latent_dim]
            """
        if layer_idx == self.num_stochastic_layers - 1:
            return self.encoder_to_obs(obs)
        else:
            return self.encoders[str(layer_idx)](previous_latent_layer)

    def get_encoder_latent_layer(self, layer_idx, previous_latent_layer=None, stop_grads=False):
        # pylint: disable=no-value-for-parameter
        """Returns distribution for single latent layer.
        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [N, batch_size] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)
        Returns: Normal distribution with event_shape [latent_dim]
            and batch_shape is [N]"""

        mu, sig = self.get_encoder_latent_layer_param(
            layer_idx, previous_latent_layer, obs=self.x)

        if stop_grads:
            mu, sig = mu.detach(), sig.detach()

        return Independent(torch.distributions.Normal(mu, sig), reinterpreted_batch_ndims=1)

    def get_inf_network(self, stop_grads=False):
        latent_dist = util.ChainDistributionFromSingle(self.get_encoder_latent_layer(
            layer_idx=self.num_stochastic_layers - 1,
            stop_grads=stop_grads))
        for layer_idx in reversed(range(self.num_stochastic_layers - 1)):
            # be careful about closures
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture/2295372
            latent_dist = util.ChainDistribution(
                latent_dist,
                lambda previous_latent_layer, layer_idx=layer_idx: self.get_encoder_latent_layer(
                    layer_idx=layer_idx,
                    previous_latent_layer=previous_latent_layer,
                    stop_grads=stop_grads))
        return util.ReversedChainDistribution(latent_dist)

    def set_internals(self, data, S=10):
        self.y = False  # VAEs are unsupervised
        self.x = data[0]
        self.inf_network = self.get_inf_network()
        if self.stop_grads:
            # self.inf_network does the sampling and self.inf_network_detached does the scoring
            self.inf_network_detached = self.get_inf_network(stop_grads=True)
        self.z = self.sample_latent(S)
        self.check_internals()

    def enable_stop_grads(self):
        #print("stop_grads activated")
        self.stop_grads = True

    def disable_stop_grads(self):
        #print("stop_grads deactivated")
        self.stop_grads = False

    def enable_reparam(self):
        #print("reparam activated")
        self.reparam = True

    def disable_reparam(self):
        #print("reparam deactivated")
        self.reparam = False


    def log_prior(self):
        """
        Returns: tensor of shape [N, S]
        """
        return self.get_decoder_latent_dist().log_prob(self.z).transpose(0, 1)

    def log_likelihood(self):
        """
        Returns: tensor of shape [N, S]
        """
        return self.get_obs_dist(self.z).log_prob(self.x).transpose(0, 1)

    def log_guide(self):
        """Log q(latent | obs).
        Args:
        Returns: tensor of shape [N, S]
        """
        assert self.inf_network is not None, 'log_guide() called before self.inf_network is set!'
        assert self.z is not None, 'log_guide() called before self.z is set!'

        if self.stop_grads:
            assert self.inf_network_detached is not None, 'using stop_grads but self.inf_network_detached is not set!'
            assert self.reparam, 'gradients not flowing through samples' #and all([z.requires_grad for z in self.z])
            return self.inf_network_detached.log_prob(self.z).transpose(0, 1)
        else:
            return self.inf_network.log_prob(self.z).transpose(0, 1)

    def get_prop_network(self):
        latent_dist = util.ChainDistributionFromSingle(
            self.get_prop_latent_layer(layer_idx=self.num_stochastic_layers - 1))
        for layer_idx in reversed(range(self.num_stochastic_layers - 1)):
            # be careful about closures
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture/2295372
            latent_dist = util.ChainDistribution(
                latent_dist,
                lambda previous_latent_layer, layer_idx=layer_idx: self.get_prop_latent_layer(
                    layer_idx=layer_idx, previous_latent_layer=previous_latent_layer))
        return util.ReversedChainDistribution(latent_dist)

    def log_prop(self):
        """Log q(latent | obs).

        Args:

        Returns: tensor of shape [N, S]
        """
        assert self.prop_network is not None, 'log_guide() called before self.inf_network is set!'
        assert self.z is not None, 'log_guide() called before self.z is set!'
        return self.prop_network.log_prob(self.z).transpose(0, 1)


    def sample_latent(self, S):
        """Samples from q(latent | obs)

        Args:
            S: int

        Returns:
            latent: tuple of tensors of shape [S, N, latent_dim].
            len(tuple) == num_stochastic_layers
        """
        assert self.inf_network is not None, 'sample_latent() called before self.inf_network is set!'
        if self.reparam:
            return self.inf_network.rsample((S, ))
        else:
            return self.inf_network.sample((S, ))


    def sample_z_proposal(self, S):
        """Samples from q(latent | obs)

        Args:
            S: int

        Returns:
            latent: tuple of tensors of shape [S, N, latent_dim].
            len(tuple) == num_stochastic_layers
        """
        assert self.prop_network is not None, 'sample_latent() called before self.inf_network is set!'
        if self.reparam:
            return self.prop_network.rsample((S, ))
        else:
            return self.prop_network.sample((S, ))


    def get_prop_latent_layer(self, layer_idx, previous_latent_layer=None):
        # pylint: disable=no-value-for-parameter
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [N, batch_size] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns: Normal distribution with event_shape [latent_dim]
            and batch_shape is [N]"""


        mu, sig = self.get_proposal_layer_param(layer_idx, previous_latent_layer, obs=self.x)
        return Independent(torch.distributions.Normal(mu, sig), reinterpreted_batch_ndims=1)

    def get_proposal_layer_param(self, layer_idx, previous_latent_layer=None, obs=None):
        # pylint: disable=not-callable,unsubscriptable-object
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [N, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [N, batch_size] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns:
            mu for Normal latent of shape [N, latent_dim]
            sig for Normal latent of shape [N, latent_dim]
            """
        return self.proposer(obs)


class ContinuousVAE(VAEBaseClass):
    def __init__(self, D, args):
        super().__init__(D, args)
        if self.args.learn_prior:
            print("Learning prior")
            self.decoder_latent_param_mu  = nn.Parameter(torch.zeros(self.latent_dims[0], device=self.args.device, dtype=torch.float))
            self.decoder_latent_param_sig = nn.Parameter(torch.ones(self.latent_dims[0], device=self.args.device, dtype=torch.float))
        else:
            self.decoder_latent_param_mu  = torch.zeros(self.latent_dims[0], device=self.args.device, dtype=torch.float)
            self.decoder_latent_param_sig = torch.ones(self.latent_dims[0], device=self.args.device, dtype=torch.float)

        self.layer_0_params = (self.decoder_latent_param_mu, self.decoder_latent_param_sig)

        self.decoders = nn.ModuleDict()
        for i in range(1, self.num_stochastic_layers):
            self.decoders[str(i)] = init_two_prong_mlp(in_dim=self.latent_dims[i - 1],
                                                    out_dim=self.latent_dims[i],
                                                    hidden_dim=self.hidden_dim,
                                                    num_layers=self.num_deterministic_layers,
                                                    non_linearity=nn.Tanh() if self.args.activation is None else  self.args.activation)

        # This is the mlp from discrete.py that doesn't produce a sigma
        self.decoder_to_obs = init_mlp(in_dim=self.latent_dims[-1],
                                       out_dim=self.D,
                                       hidden_dim=self.hidden_dim,
                                       num_layers=self.num_deterministic_layers,
                                       non_linearity=nn.Tanh() if self.args.activation is None else  self.args.activation)


        self.encoder_to_obs = init_two_prong_mlp(in_dim=self.D,
                                            out_dim=self.latent_dims[-1],
                                            hidden_dim=self.hidden_dim,
                                            num_layers=self.num_deterministic_layers,
                                            non_linearity=nn.Tanh() if self.args.activation is None else  self.args.activation)

        self.encoders = nn.ModuleDict()
        for i in reversed(range(self.num_stochastic_layers - 1)):
            self.encoders[str(i)] = init_two_prong_mlp(in_dim=self.latent_dims[i + 1],
                                                    out_dim=self.latent_dims[i],
                                                    hidden_dim=self.hidden_dim,
                                                    num_layers=self.num_deterministic_layers,
                                                    non_linearity=nn.Tanh() if self.args.activation is None else  self.args.activation)
