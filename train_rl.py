import gym
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList

from agents.rl_birdview.utils.wandb_callback import WandbCallback
from gym_navsim.utils import config_utils
import warnings

log = logging.getLogger(__name__)

#warnings.filterwarnings("error")
@hydra.main(config_path='config', config_name='train_rl')
def main(cfg: DictConfig):
    # prepare agent
    agent_name = cfg.actors[cfg.ev_id].agent

    last_checkpoint_path = Path(hydra.utils.get_original_cwd()) / 'outputs' / 'checkpoint.txt'
    if last_checkpoint_path.exists():
        with open(last_checkpoint_path, 'r') as f:
            cfg.agent[agent_name].wb_run_path = f.read()

    OmegaConf.save(config=cfg.agent[agent_name], f='config_agent.yaml')

    # single agent
    AgentClass = config_utils.load_entry_point(cfg.agent[agent_name].entry_point)
    agent = AgentClass('config_agent.yaml')
    cfg_agent = OmegaConf.load('config_agent.yaml')

    obs_configs = {cfg.ev_id: OmegaConf.to_container(cfg_agent.obs_configs)}
    reward_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].reward)}
    terminal_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].terminal)}


    # env wrapper
    EnvWrapper = config_utils.load_entry_point(cfg_agent.env_wrapper.entry_point)
    wrapper_kargs = cfg_agent.env_wrapper.kwargs

    def env_maker(config):
        env = gym.make(config['env_id'],token=cfg.token, obs_configs=obs_configs, reward_configs=reward_configs,
                       terminal_configs=terminal_configs,**config['env_configs'])
        env = EnvWrapper(env, **wrapper_kargs)
        return env
    envs = OmegaConf.to_container(cfg.train_envs)["envs"]
    if cfg.dummy:
        env = DummyVecEnv([lambda config=config: env_maker(config) for config in envs])
    else:
        env = SubprocVecEnv([lambda config=config: env_maker(config) for config in envs])
    env.render_mode = "rgb_array"
    # wandb init
    wb_callback = WandbCallback(cfg, env)
    callback = CallbackList([wb_callback])

    # save wandb run path to file such that bash file can find it
    with open(last_checkpoint_path, 'w') as f:
        f.write(wandb.run.path)
    
    agent.learn(env, total_timesteps=int(cfg.total_timesteps), callback=callback, seed=cfg.seed)
    #server_manager.stop()
if __name__ == '__main__':
    main()
    log.info("train_rl.py DONE!")