from gym_navsim.core.obs_manager.obs_manager import ObsManagerBase
from gym import spaces
import numpy as np

class ObsManager(ObsManagerBase):
    def __init__(self, obs_config):
        super(ObsManager, self).__init__()
    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'speed_xy': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'forward_speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        })
    def attach_ego_vehicle(self, ego_vehicle):
        self.scene = ego_vehicle.scene
        self.agent_input = ego_vehicle.agent_input
    def get_observation(self,timestamp):
        velocity = self.agent_input.ego_velocity
        np_vel = np.array([velocity[0],velocity[1],0]) # z - axis is 0 for now 
        np_fvec = np.array([1, 0, 0]) # Car is looking at top at the start

        speed = np.linalg.norm(np_vel)
        speed_xy = np.linalg.norm(np_vel[0:2])
        forward_speed = np.dot(np_vel, np_fvec)

        obs = {
            'speed': np.array([speed], dtype=np.float32),
            'speed_xy': np.array([speed_xy], dtype=np.float32),
            'forward_speed': np.array([forward_speed], dtype=np.float32)
        }
        return obs   