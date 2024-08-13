import gym
import json
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os.path
import sys

from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from gym_navsim.utils import config_utils
from agents.rl_birdview.utils.wandb_callback import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
from navsim.common.dataclasses import Trajectory, TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import pandas as pd

log = logging.getLogger(__name__)

def calculate_pdm_score(trajectory,metric_cache):
    initial_ego_state = metric_cache.ego_state
    sampling = TrajectorySampling(num_poses=8,interval_length=0.5)
    if len(trajectory) < 12:
        # Kaza yapmış
        return {
            "nac": 0,
            "dac": 0,
            "ddc": 0,
            "ep": 0,
            "ttc": 0,
            "c": 0,
            "score": 0
        }
    trajectory = Trajectory(poses=trajectory[4:],trajectory_sampling=sampling)
    pred_trajectory = transform_trajectory(trajectory,initial_ego_state)
    pdm_trajectory = metric_cache.trajectory
    future_sampling = TrajectorySampling(num_poses=40,interval_length=0.1)
    pdm_states, pred_states = (
            get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
            get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
        )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
    simulator = PDMSimulator(future_sampling)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer = PDMScorer(future_sampling)
    scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
    pred_idx = 1

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]
    driving_direction_compliance = scorer._multi_metrics[
        MultiMetricIndex.DRIVING_DIRECTION, pred_idx
    ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]

    score = scores[pred_idx]
    pdm_score = {
        "nac": no_at_fault_collisions,
        "dac": drivable_area_compliance,
        "ddc": driving_direction_compliance,
        "ep": ego_progress,
        "ttc": time_to_collision_within_bound,
        "c": comfort,
        "score": score
    }
    return pdm_score
def run_single(token, env, agent, agents_log_dir, log_video, max_step=None):
    list_render = []
    ep_stat_dict = {}
    ep_event_dict = {}
    log_dir = agents_log_dir / "ppo"
    log_dir.mkdir(parents=True, exist_ok=True)
    agent.reset(log_dir / f'{token}.log')
    
    log.info(f'Start Benchmarking {token}.')
    actor_id = "hero"
    env.env_method("set_scene",token)
    obs = env.reset()
    # Çok çirkin şu an ama olsun
    wrapper = env.envs[0].gym_env
    timestamp = wrapper.time
    done = False
    ev = env.get_attr("ev_handler")[0].ego_vehicles["hero"]
    while not done:
        actions, values, log_probs, mu, sigma, _ = agent._policy.forward(obs)

        new_obs, reward, done, info = env.step(actions)

        render_imgs = []
        if log_video:
            render_imgs.append(wrapper.render())
        if done and (actor_id not in ep_stat_dict):
            ep_stat_dict[actor_id] = info[0]['episode_stat']
            ep_event_dict[actor_id] = info[0]['episode_event']

        if len(list_render) > 15000:
            del list_render[0]
        if log_video:
            list_render.append(tile_images(render_imgs))

        timestamp = wrapper.time
        if max_step and timestamp > max_step:
            break
        obs = new_obs
    # TODO : Bunu daha güzel yazmam lazım
    trajectory = ev.trajectory
    metric_cache = ev.metric_cache
    pdm_score = calculate_pdm_score(trajectory,metric_cache)
    return list_render, pdm_score


@hydra.main(config_path='config', config_name='benchmark')
def main(cfg: DictConfig):
    log.setLevel(getattr(logging, cfg.log_level.upper()))
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
                       terminal_configs=terminal_configs,benchmark=True,**config['env_configs'])
        env = EnvWrapper(env,benchmark=True,**wrapper_kargs)
        return env
    envs = OmegaConf.to_container(cfg.train_envs)["envs"]
    if cfg.dummy:
        env = DummyVecEnv([lambda config=config: env_maker(config) for config in envs])
    else:
        env = SubprocVecEnv([lambda config=config: env_maker(config) for config in envs])

    log.info(f"Start Benchmarking! agent_name: {agent_name},envs: {envs}")

    # make directories
    diags_dir = Path('diagnostics') / "benchmark"
    agents_log_dir = Path('agents_log') / "benchmark"
    video_dir = Path('videos') / "benchmark"
    diags_dir.mkdir(parents=True, exist_ok=True)
    agents_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=["token","nac","dac","ddc","ep","ttc","c","score"])

    # init wandb
    wandb.init(project=cfg.wb_project, name="benchmark", group=cfg.wb_group, notes=cfg.wb_notes, tags=cfg.wb_tags)
    wandb.config.update(OmegaConf.to_container(cfg))
    wandb.save('./config_agent.yaml')

    for i,token in enumerate(env.envs[0].scene_loader.tokens):
        log.info(f"Start Benchmarking env_idx {i}, token: {token}")
        run_name = f"{token}"
        if i == 10:
            break
        list_render, pdm_score = run_single(
            run_name, env, agent, agents_log_dir, cfg.log_video)

        # log video
        if cfg.log_video:
            video_path = (video_dir / f'{run_name}.mp4').as_posix()
            encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
            for im in list_render:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None
            wandb.log({f'video/-{run_name}': wandb.Video(video_path)})
        df = pd.concat([df,pd.DataFrame([{"token":run_name,**pdm_score}])])
    df.to_csv("pdm_scores.csv",index=False)
    """
    # dump events
    diags_json_path = (diags_dir / f'{run_name}.json').as_posix()
    with open(diags_json_path, 'w') as fd:
        json.dump(ep_event_dict, fd, indent=4, sort_keys=False)

    # save diags and agents_log
    wandb.save(diags_json_path)
    """

    """
    # save statistics
    for actor_id, ep_stat in ep_stat_dict.items():
        ep_stat_buffer[actor_id].append(ep_stat)
        log_dict = {}
        for k, v in ep_stat.items():
            k_actor = f'{actor_id}/{k}'
            log_dict[k_actor] = v
        wandb.log(log_dict, step=task_idx)
    
    with open(ep_state_buffer_json, 'w') as fd:
        json.dump(ep_stat_buffer, fd, indent=4, sort_keys=True)
    """
    # clean up
    list_render.clear()
    ep_stat_dict = None
    ep_event_dict = None

    # close env
    env.close()
    env = None
    """
    # log after suite is completed
    table_data = []
    ep_stat_keys = None
    for actor_id, list_ep_stat in json.load(open(ep_state_buffer_json, 'r')).items():
        avg_ep_stat = WandbCallback.get_avg_ep_stat(list_ep_stat)
        data = [suite_name, actor_id, str(len(list_ep_stat))]
        if ep_stat_keys is None:
            ep_stat_keys = list(avg_ep_stat.keys())
        data += [f'{avg_ep_stat[k]:.4f}' for k in ep_stat_keys]
        table_data.append(data)

    table_columns = ['Suite', 'actor_id', 'n_episode'] + ep_stat_keys
    wandb.log({'table/summary': wandb.Table(data=table_data, columns=table_columns)})

    with open(last_checkpoint_path, 'w') as f:
        f.write(f'{env_idx+1}')

    log.info(f"Finished Benchmarking env_idx {env_idx}, suite_name: {suite_name}")
    if env_idx+1 == len(cfg.test_suites):
        log.info(f"Finished, {env_idx+1}/{len(cfg.test_suites)}")
        return
    else:
        log.info(f"Not finished, {env_idx+1}/{len(cfg.test_suites)}")
        sys.exit(1)
    """

if __name__ == '__main__':
    main()
    log.info("data_collect.py DONE!")
