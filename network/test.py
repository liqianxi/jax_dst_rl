import metaworld
import random
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax

mt10 = metaworld.MT10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in mt10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in mt10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

# for env in training_envs:
#   obs = env.reset()  # Reset environment
#   a = env.action_space.sample()  # Sample an action
#   obs, reward, terminated, truncated, info = env.step(a)  # Step the environment with the sampled random action
# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# import optax
# from flax.linen.initializers import constant, orthogonal
# from typing import Sequence, NamedTuple, Any
# from flax.training.train_state import TrainState
# import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

env, env_params = BraxGymnaxWrapper(mt10.train_classes["pick-place-v2"]), None
env = LogWrapper(env)
env = ClipAction(env)
env = VecEnv(env)

