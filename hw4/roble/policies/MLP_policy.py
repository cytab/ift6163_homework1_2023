import abc
import itertools
import numpy as np
import torch
import hw4.roble.util.class_util as classu

from hw4.roble.infrastructure import pytorch_util as ptu
from hw4.roble.policies.base_policy import BasePolicy
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    @classu.hidden_member_initialize
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 deterministic=False,
                 learn_policy_std=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           n_layers=self._n_layers,
                                           size=self._size)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._learn_policy_std = learn_policy_std
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim, 
                                      output_size=self._ac_dim,
                                      n_layers=self._n_layers, size=self._size)
            self._mean_net.to(ptu.device)
            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.2
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )
        self.nn_baseline = nn_baseline
        if nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                n_layers=self._n_layers,
                size=self._size,
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        action = self.forward(obs)
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                ##  TODO output for a deterministic policy
                logit = self._mean_net(observation)
                action_distribution = logit
            else:
                batch_mean = self._mean_net(observation)
                scale_tril = torch.diag(torch.exp(self._std))
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )
        return action_distribution

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        obs = ptu.from_numpy(observations)
        action = self.forward(obs)
        
        loss = q_fun.q_net(obs, action)
        
        actor_loss = -loss.mean()
        self._optimizer.zero_grad()
        actor_loss.backward()
        self._optimizer.step()
        
        return actor_loss.item()
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution
        dist = self.forward(ptu.from_numpy(obs))
        action = dist.rsample()
        log = dist.log_prob(action)
        return ptu.to_numpy(action), log
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff
        obs = ptu.from_numpy(observations)

        actions_distribution = self(obs)
        actions = actions_distribution.rsample()
        
        q_values = q_fun.q_net(obs, actions)
        q_values = q_values.detach()

        log_prob = actions_distribution.log_prob(actions)
        entropy = actions_distribution.entropy()

        loss = -(log_prob * q_values - self.entropy_coeff * entropy).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        
        return loss.item()
    
class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, learn_policy_std=False, **kwargs):
        
        super().__init__(ac_dim, ob_dim, n_layers, size, learn_policy_std=False, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        if self.nn_baseline:
            q_values_norm = (q_values - q_values.mean()) / (q_values.std() + 1e-8)
            baseline_targets = ptu.from_numpy(q_values_norm)
            
            self._baseline_optimizer.zero_grad() 
            baseline_loss = self.baseline_loss(self._baseline(observations).squeeze(), baseline_targets)
            baseline_loss.backward()
            self._baseline_optimizer.step()

        self._optimizer.zero_grad()
        loss = -(self(observations).log_prob(actions) * (advantages)).mean()
        loss.backward()
        self._optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }

        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self._baseline(observations)
        return ptu.to_numpy(pred.squeeze())
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution
        dist = self.forward(ptu.from_numpy(obs))
        action = dist.rsample()
        return ptu.to_numpy(action)