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
