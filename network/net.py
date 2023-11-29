import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from .distribution import TanhNormal
import torch.nn.functional as F
import torchrl.networks.init as init
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20



# class EmbedGuassianContPolicy(nn.Module):
#     def forward(self, x, neuron_masks,enable_mask=True):

#         x = super().forward(x, neuron_masks,enable_mask)

#         mean, log_std = x.chunk(2, dim=-1)

#         log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#         std = torch.exp(log_std)

#         return mean, std, log_std

#     def eval_act(self, x, neuron_masks):
#         with torch.no_grad():
#             mean, _, _ = self.forward(x, neuron_masks)
#         return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

#     def explore( self, x, neuron_masks, enable_mask=True,return_log_probs = False, return_pre_tanh = False ):

#         mean, std, log_std = self.forward(x, neuron_masks, enable_mask)

#         dis = TanhNormal(mean, std)

#         ent = dis.entropy().sum(-1, keepdim=True) 

#         dic = {
#             "mean": mean,
#             "log_std": log_std,
#             "ent":ent
#         }

#         if return_log_probs:
#             action, z = dis.rsample(return_pretanh_value=True)
#             log_prob = dis.log_prob(
#                 action,
#                 pre_tanh_value=z
#             )
#             log_prob = log_prob.sum(dim=-1, keepdim=True)
#             dic["pre_tanh"] = z.squeeze(0)
#             dic["log_prob"] = log_prob
#         else:
#             if return_pre_tanh:
#                 action, z = dis.rsample(return_pretanh_value=True)
#                 dic["pre_tanh"] = z.squeeze(0)
#             action = dis.rsample(return_pretanh_value=False)

#         dic["action"] = action.squeeze(0)
#         return dic

#     def update(self, obs, actions):
#         #TODO: to be updated.
#         mean, std, log_std = self.forward(obs)
#         dis = TanhNormal(mean, std)

#         log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
#         ent = dis.entropy().sum(-1, keepdim=True) 
        
#         out = {
#             "mean": mean,
#             "log_std": log_std,
#             "log_prob": log_prob,
#             "ent": ent
#         }
#         return out



class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)
 
