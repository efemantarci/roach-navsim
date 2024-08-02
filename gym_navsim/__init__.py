from gym.envs.registration import register
from .navsim_multi_agent_env import NavsimEnv
register(id="navsim-v0", entry_point=NavsimEnv)