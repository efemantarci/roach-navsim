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
    def __init__(self,token,obs_configs,reward_configs,terminal_configs,render_mode="human"):
        self.token = token
        self.render_mode = render_mode
        self.scene_loader = self._generate_scene_loader()
        self._initialize_scene(token)
        self._om_handler = ObsManagerHandler(obs_configs)
        self._ev_handler = EgoVehicleHandler(reward_configs,terminal_configs)
        self.observation_space = self._om_handler.observation_space
        self._obs_configs = obs_configs
        self.action_space = gym.spaces.Dict({ego_vehicle_id: gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)
            for ego_vehicle_id in obs_configs.keys()})
        """
        self.action_space = gym.spaces.Dict({ego_vehicle_id: gym.spaces.Tuple(
            [gym.spaces.Box(
                low=np.array([-32.0, -32.0, -np.pi]),
                high=np.array([32.0, 32.0, np.pi]),
                dtype=np.float32) for _ in range(8)])
            for ego_vehicle_id in obs_configs.keys()})
        """
        self.time = 0
    def _generate_scene_loader(self):
        
        SPLIT = "test"
        FILTER = hydra.utils.get_original_cwd() + "/config/scene_filter/navtest.yaml"
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
    def _initialize_scene(self,token):
        self.scene = self.scene_loader.get_scene_from_token(token)
    def reset(self):
        self.time = 0
        new_token = np.random.choice(self.scene_loader.tokens)
        #new_token = "46deeff0d0495df6"
        self._initialize_scene(new_token)
        self._ev_handler.reset(self._obs_configs,self.scene)
        logger.debug("_ev_handler reset done!!")
        self._om_handler.reset(self._ev_handler.ego_vehicles)
        logger.debug("_om_handler reset done!!")
        # Burada environment tick atılmalı
        obs_dict = self._om_handler.get_observation(self.time)
        
        return obs_dict
    def step(self,control_dict):
        self._ev_handler.apply_control(control_dict)
        self.time += 1
        obs_dict = self._om_handler.get_observation(self.time)
        reward_dict, done_dict, info_dict = self._ev_handler.tick(self.time)
        return (obs_dict, reward_dict, done_dict, info_dict)
    def render(self):
        if self.render_mode == "human":
            frame_idx = self.scene.scene_metadata.num_history_frames - 1  # current frame
            fig, ax = plot_bev_frame(self.scene, frame_idx)
            plt.show()
        return "rendering"