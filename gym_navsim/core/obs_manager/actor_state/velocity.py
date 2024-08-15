import numpy as np
from gym import spaces

from gym_navsim.core.obs_manager.obs_manager import ObsManagerBase
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        # acc_x, acc_y: m/s2
        # vel_x, vel_y: m/s
        # vel_angular z: rad/s
        self.obs_space = spaces.Dict({
            'acc_xy': spaces.Box(low=-1e3, high=1e3, shape=(2,), dtype=np.float32),
            'vel_xy': spaces.Box(low=-1e2, high=1e2, shape=(2,), dtype=np.float32),
            'vel_ang_z': spaces.Box(low=-1e3, high=1e3, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.scene = ego_vehicle.scene
        self.agent_input = self.scene.get_agent_input().ego_statuses[-1]

    def get_observation(self,timestamp):
        state = self.ego_vehicle.states[timestamp]
        acc_w = 0
        vel_w = [state[3]]
        ang_w = 0
        self.ego_vehicle.velocity = vel_w
        # acc_w = np.array([state[StateIndex.ACCELERATION_X], state[StateIndex.ACCELERATION_Y]])
        # vel_w = np.array([state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y]])
        # self.ego_vehicle.velocity = vel_w
        # ang_w = np.array([state[StateIndex.ANGULAR_VELOCITY]])
        #ang_w = self._parent_actor.vehicle.get_angular_velocity() We don't have angular velocity for now

        obs = {
            'acc_xy': np.array(acc_w, dtype=np.float32),
            'vel_xy': np.array(vel_w, dtype=np.float32),
            'vel_ang_z': np.array(ang_w, dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
