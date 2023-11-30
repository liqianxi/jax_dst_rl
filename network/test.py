
from gymnasium.spaces import Box
import numpy as np


from metaworld import MT10
# import numpy as np
# import random
import jax
import jax.numpy as jnp
# import flax.linen as nn
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
# import optax
# from flax.linen.initializers import constant, orthogonal
# from typing import Sequence, NamedTuple, Any
# from flax.training.train_state import TrainState
# import distrax
# import gymnax
def make_train(config):
    mt10 = MT10() # Construct the benchmark, sampling tasks



    env = mt10.train_classes["pick-place-v2"]
    env, env_params = BraxGymnaxWrapper(env), None

    
    # env = LogWrapper(env)
    # env = ClipAction(env)
    #env = VecEnv(env)
    # batched_seeds = jnp.array([1, 2, 3, 4])
    # batched_options = jnp.array([{'param': 0}, {'param': 1}, {'param': 2}, {'param': 3}])
    # obs = env.reset(batched_seeds,batched_options)  # Reset environment
    # a = env.action_space.sample()  # Sample an action
    # obs, reward, terminated, truncated, info = env.step(a)  # Step the environment with the sampled random action

    # print(obs, reward, terminated, truncated, info)
if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)