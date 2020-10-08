from spinup.algos.pytorch.ppo.ppo import ppo as ppo
from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.core import MLPCritic
from spinup.algos.pytorch.ppo.core import Actor
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import maze

import os

import numpy as np
import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class DummyDistribution(object):
    """
    Spinning Up algorithm relies on a few methods of Distribution.
    It should implement at least: entropy(), log_prob(), sample().
    """
    
    def __init__(self, logits):
        self.dist = Categorical(logits=logits)
    
    def entropy(self):
        return self.dist.entropy()
    
    def log_prob(self, value):
        return self.dist.log_prob(value)
    
    def sample(self):
        return self.dist.sample()

class CustomActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return DummyDistribution(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MazeActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(32,32), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        #if isinstance(action_space, Box):
        #    self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        #elif isinstance(action_space, Discrete):
        self.pi = CustomActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

"""
class PolicyNetwork(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    
    def _distribution(self, obs):
        pass
    
    def _log_prob_from_distribution(self, pi, act):
        pass

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def sample(self):
        pass
    
    def log_prob(self, act):
        pass
"""

# pi:
#  pi = self.pi._distribution(obs)
#   _log_prob_from_distribution(pi, a)
#  pi.sample()
#  pi.log_prob(act)
# 
##

if __name__ == '__main__':
    
    #activation=nn.Tanh
    #a = MLPCategoricalActor(3,2,[4,5],activation=nn.Tanh)
    
    env_fn = maze.MazeEnv

    exp_name = 'maze-x'

    logger_kwargs = dict(
        exp_name=exp_name,
        output_dir='log_{}'.format(exp_name)
    )
    
    actor_critic = MazeActorCritic
    
    # def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
    #         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    #         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
    #         target_kl=0.01, logger_kwargs=dict(), save_freq=10)
    ppo(env_fn=env_fn, actor_critic=actor_critic,
        logger_kwargs=logger_kwargs)
