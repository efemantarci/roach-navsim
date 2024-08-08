import lzma
import pickle
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
import numpy as np
import os
from shapely import affinity
from shapely.geometry import Polygon, LineString
from nuplan.common.actor_state.state_representation import TimePoint
from gym_navsim.utils.conversion import convert_absolute_to_relative_se2_array
from nuplan.common.geometry.convert import absolute_to_relative_poses,relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2
class EgoVehicle:
    def __init__(self,scene) -> None:
        self.scene = scene
        self.agent_input = scene.get_agent_input().ego_statuses[-1]
        self.collision_px = False
        # Metric cache
        metadata = self.scene.scene_metadata
        # This will change
        metric_cache_path = os.path.join(os.environ.get("NAVSIM_EXP_ROOT"),"public_test_metric_cache",metadata.log_name,"unknown",metadata.initial_token,"metric_cache.pkl")
        f = lzma.open(metric_cache_path,"rb")
        self.metric_cache = pickle.load(f)
        # Route info
        initial_ego_state = self.metric_cache.ego_state
        pdm_trajectory = self.metric_cache.trajectory
        future_sampling = TrajectorySampling(num_poses=8,interval_length=0.5)
        pdm_states = get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point)[:,:3]
        self.route = convert_absolute_to_relative_se2_array(initial_ego_state.rear_axle,pdm_states)
        self.route_abs = [np.array([*x]) for x in pdm_states]
        self.token = scene.scene_metadata.initial_token
        self.pdm_score = {
            "nac": 1,
            "dac": 1,
            "ddc": 1,
            "ep": 1,
            "ttc": 1,
            "c": 1,
            "terminal_reward": 0
        }
        self.time = 0
        self.steer = 0
        self.velocity = None

        past_poses = np.array([ego_status.ego_pose for ego_status in self.scene.get_agent_input().ego_statuses])
        human_poses = np.array(self.scene.get_future_trajectory().poses)
        self.human_trajectory = np.concatenate([past_poses,human_poses])
        self.trajectory = past_poses
        self.past_poses = past_poses
    def rotate(self,points, angle):
        """Rotate points by a given angle."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix.T