import logging
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame
from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from omegaconf import OmegaConf
import hydra

logger = logging.getLogger(__name__)
class NavsimEnv(gym.Env):
    def __init__(self,token,obs_configs,reward_configs,terminal_configs,benchmark=False,render_mode="rgb_array"):
        self.token = token
        self.render_mode = render_mode
        self.test_scene_loader = self._generate_scene_loader("test")
        self.benchmark = benchmark
        if not benchmark:
            self.split = "trainval"
            self.train_scene_loader = self._generate_scene_loader("trainval")
            self.scene_loader = self.train_scene_loader
        else:
            self.split = "test"
            self.scene_loader = self.test_scene_loader
            self.token_idx = 0
            self.tokens = self.scene_loader.tokens
        self.initialize_scene(token)
        self.om_handler = ObsManagerHandler(obs_configs)
        self.ev_handler = EgoVehicleHandler(reward_configs,terminal_configs)
        self.observation_space = self.om_handler.observation_space
        self._obs_configs = obs_configs
        self.action_space = gym.spaces.Dict({ego_vehicle_id: gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)
            for ego_vehicle_id in obs_configs.keys()})
        self.time = 0
    def _generate_scene_loader(self,split):
        SPLIT = split
        if SPLIT == "test":
            FILTER = hydra.utils.get_original_cwd() + "/config/scene_filter/navtest.yaml"
        elif SPLIT == "trainval":
            FILTER = hydra.utils.get_original_cwd() + "/config/scene_filter/navtrain.yaml"
        cfg = OmegaConf.load(FILTER)
        scene_filter = instantiate(cfg)
        openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

        self.scene_loader = SceneLoader(
            openscene_data_root / f"navsim_logs/{SPLIT}",
            openscene_data_root / f"sensor_blobs/{SPLIT}",
            scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        return self.scene_loader
    def initialize_scene(self,token):
        self.scene = self.scene_loader.get_scene_from_token(token)
    def reset(self):
        self.time = 0
        if not self.benchmark:
            new_token = np.random.choice(self.scene_loader.tokens)
            #new_token = "4a6ab6e35934543a" #"46deeff0d0495df6"
            self.initialize_scene(new_token)
        self.ev_handler.reset(self._obs_configs,self.scene,self.split)
        logger.debug("ev_handler reset done!!")
        self.om_handler.reset(self.ev_handler.ego_vehicles)
        logger.debug("om_handler reset done!!")
        # Burada environment tick aılmalı
        obs_dict = self.om_handler.get_observation(self.time)
        
        return obs_dict
    def step(self,control_dict):
        self.ev_handler.apply_control(control_dict)
        self.time += 1
        obs_dict = self.om_handler.get_observation(self.time)
        reward_dict, done_dict, info_dict = self.ev_handler.tick(self.time)
        return (obs_dict, reward_dict, done_dict, info_dict)
    def render(self):
        if self.render_mode == "human":
            frame_idx = self.scene.scene_metadata.num_history_frames - 1  # current frame
            fig, ax = plot_bev_frame(self.scene, frame_idx)
            plt.show()
        return "rendering"
    def switch_scene_loader(self,split):
        self.split = split
        if split == "trainval":
            self.scene_loader = self.train_scene_loader
        elif split == "test":
            self.scene_loader = self.test_scene_loader