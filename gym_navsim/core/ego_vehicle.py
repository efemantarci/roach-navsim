import lzma
import pickle
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
import numpy as np
import os
from nuplan.common.actor_state.state_representation import TimePoint
from gym_navsim.utils.conversion import convert_absolute_to_relative_se2_array
from shapely import Point
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from gym_navsim.utils.bicycle_model import KinematicBicycleModel
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
class EgoVehicle:
    def __init__(self,scene,split) -> None:
        self.scene = scene
        self.agent_input = scene.get_agent_input().ego_statuses[-1]
        self.collision_px = False
        self.outside_road = False
        self.run_rl = False
        # Metric cache
        metadata = self.scene.scene_metadata
        # This will change
        if split == "trainval":
            filename = "public_navtrain_metric_cache"
        elif split == "test":
            filename = "public_test_metric_cache"
        metric_cache_path = os.path.join(os.environ.get("NAVSIM_EXP_ROOT"),filename,metadata.log_name,"unknown",metadata.initial_token,"metric_cache.pkl")
        f = lzma.open(metric_cache_path,"rb")
        self.metric_cache = pickle.load(f)
        # Route info
        initial_ego_state = self.metric_cache.ego_state
        self.commands = []
        self.states = [ego_state_to_state_array(initial_ego_state)]
        inital_state = ego_state_to_state_array(initial_ego_state)
        # x,y,heading,v_x,a_x,steering_angle
        first_state = np.array([0,0,0,inital_state[StateIndex.VELOCITY_X],inital_state[StateIndex.ACCELERATION_X],inital_state[StateIndex.STEERING_ANGLE]])
        self.states.append(first_state)
        # Absolute yerine relative koordinat istiyorum
        self.states[0][:3] = 0
        pdm_trajectory = self.metric_cache.trajectory
        start_time = pdm_trajectory.start_time.time_us
        times = [TimePoint(time) for time in np.linspace(start_time,start_time + 8 * 0.5 * 1e6,9)]
        pdm_states = pdm_trajectory.get_state_at_times(times)
        self.route = convert_absolute_to_relative_se2_array(initial_ego_state.rear_axle, np.array([[*se2.center] for se2 in pdm_states]))[1:]
        self.route_abs = np.array([[*se2.center] for se2 in pdm_states])
        # Added to fit the centerline
        center_linestring = self.metric_cache.centerline.linestring
        first_pdm_point = Point(self.route_abs[0][0],self.route_abs[0][1])
        ego_start_point = Point(initial_ego_state.center.x,initial_ego_state.center.y)
        closest_point = center_linestring.interpolate(center_linestring.project(first_pdm_point))
        difference = np.array([first_pdm_point.x - closest_point.x,first_pdm_point.y - closest_point.y])
        self.route_abs[:,:2] -= difference
        self.token = scene.scene_metadata.initial_token
        self.pdm_score = {
            "nac": 1,
            "dac": 1,
            "ddc": 1,
            "ep": 1,
            "ttc": 1,
            "c": 1,
            "terminal_reward": 0,
            "score":1
        }
        self.time = 0
        self.steer = 0
        self.velocity = None

        past_poses = np.array([ego_status.ego_pose for ego_status in self.scene.get_agent_input().ego_statuses])
        human_poses = np.array(self.scene.get_future_trajectory().poses)
        self.human_trajectory = np.concatenate([past_poses,human_poses])
        self.trajectory = past_poses
        self.past_poses = past_poses
        wheelbase = get_pacifica_parameters().wheel_base
        self.motion_model = KinematicBicycleModel(wheelbase=wheelbase,delta_time=0.5,max_steer=np.pi/3)