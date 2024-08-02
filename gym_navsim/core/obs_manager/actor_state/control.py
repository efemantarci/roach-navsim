import numpy as np
from gym import spaces

from gym_navsim.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):
        super(ObsManager, self).__init__()
        self.ego_vehicle = None

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'throttle': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'steer': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'brake': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'gear': spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),  # 0-5
            'speed_limit': spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)
        })

    def get_observation(self,timestamp):
        """
        control = self._parent_actor.vehicle.get_control()
        speed_limit = self._parent_actor.vehicle.get_speed_limit() / 3.6 * 0.8
        obs = {
            'throttle': np.array([control.throttle], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'gear': np.array([control.gear], dtype=np.float32),
            'speed_limit': np.array([speed_limit], dtype=np.float32),
        }
        return obs
        Bunu şu an yapamıyorum
        """
        obs = {
            'throttle': np.array([0], dtype=np.float32),
            'steer': np.array([0], dtype=np.float32),
            'brake': np.array([0], dtype=np.float32),
            'gear': np.array([0], dtype=np.float32),
            'speed_limit': np.array([0], dtype=np.float32),
        }
        return obs
    def attach_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
    def clean(self):
        self._parent_actor = None
