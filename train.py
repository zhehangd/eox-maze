import os

import numpy as np
import torch
import torch.nn as nn

from spinup.algos.pytorch.ppo.ppo import ppo as ppo

import ac
import maze

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
    
    actor_critic = ac.MazeActorCritic
    
    # def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
    #         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    #         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
    #         target_kl=0.01, logger_kwargs=dict(), save_freq=10)
    ppo(env_fn=env_fn, actor_critic=actor_critic,
        logger_kwargs=logger_kwargs, epochs=200)
