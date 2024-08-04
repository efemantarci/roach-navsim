import numpy as np
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
from navsim.common.dataclasses import Trajectory, TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
from navsim.common.enums import BoundingBoxIndex
class ValeoAction(object):

    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.scene = ego_vehicle.scene
        self.metric_cache = ego_vehicle.metric_cache
        self._last_steer = 0
        self._maximum_speed = 6.0

    def get(self, terminal_reward):
        calculate_trajectory = self.ego_vehicle.trajectory[4:] # Origin is not included in trajectory
        self.traj = calculate_trajectory # Şimdilik fonksiyona böyle vericem sonra hallederim
        reward_debug = self.update_pdm_score(terminal_reward)

        if abs(self.ego_vehicle.steer - self._last_steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        self._last_steer = self.ego_vehicle.steer

        rotated_pdm_points = self.convert_relative_trajectories(self.ego_vehicle.route[:,:3],self.ego_vehicle.time - 1,calculate_trajectory,self.ego_vehicle.human_trajectory,3)
        d_vec = calculate_trajectory[-1,:2] - rotated_pdm_points[self.ego_vehicle.time - 1,:2]
        unit_right = np.array([0,1])
        lateral_distance = np.abs(np.dot(d_vec,unit_right))
        r_position = -1.0 * (lateral_distance / 2.0)

        angle_difference = np.abs(calculate_trajectory[-1,2] - rotated_pdm_points[self.ego_vehicle.time - 1,2]) / np.pi
        r_rotation = -1.0 * angle_difference

        desired_spd_veh = self._maximum_speed
        start_idx = self.scene.scene_metadata.num_history_frames + self.ego_vehicle.time - 1 - 1 # -1 comes because we are rewarding previous frame
        hazard_vehicle_loc = self.lbc_hazard_vehicle(self.scene.frames[start_idx],proximity_threshold=9.5)
        if hazard_vehicle_loc is not None:
            print("Hazard vehicle detected",hazard_vehicle_loc)
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2])-8.0)
            desired_spd_veh = self._maximum_speed * np.clip(dist_veh, 0.0, 5.0)/5.0

        desired_speed = min(desired_spd_veh, self._maximum_speed)
        ev_speed = np.linalg.norm(self.ego_vehicle.velocity)

        r_speed = 1.0 - np.abs(ev_speed - desired_speed) / self._maximum_speed

        reward_debug["debug_texts"].append(f'r_position:{r_position:5.2f}')
        reward_debug["debug_texts"].append(f'r_rotation:{r_rotation:5.2f}')
        reward_debug["debug_texts"].append(f'r_action:{r_action:5.2f}')
        reward_debug["debug_texts"].append(f'r_speed:{r_speed:5.2f}')

        #reward = self.ego_vehicle.pdm_score["ep"] + terminal_reward + r_position + r_rotation + r_action
        reward = terminal_reward + r_position + r_rotation + r_action + r_speed
        # Ego progress yaptım şimdilik
        return reward, reward_debug
    def _convert_to_trajectory(self,poses):
        future_sampling = TrajectorySampling(num_poses=len(poses), interval_length=0.5)
        return Trajectory(poses=poses, trajectory_sampling=future_sampling)
    def update_pdm_score(self,terminal_reward):
        calculate_len = len(self.traj)
        initial_ego_state = self.metric_cache.ego_state
        pred_trajectory = transform_trajectory(self._convert_to_trajectory(self.traj),initial_ego_state)
        pdm_trajectory = self.metric_cache.trajectory
        future_sampling = TrajectorySampling(num_poses=5 * calculate_len,interval_length=0.1)
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
                self.metric_cache.observation,
                self.metric_cache.centerline,
                self.metric_cache.route_lane_ids,
                self.metric_cache.drivable_area_map,
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
        debug_texts = [
            f'pdm:{score:5.2f}',
            f'nac:{no_at_fault_collisions:5.2f}',
            f'dac:{drivable_area_compliance:5.2f}',
            f'ddc:{driving_direction_compliance:5.2f}',
            f'ep:{ego_progress:5.2f}',
            f'ttc:{time_to_collision_within_bound:5.2f}',
            f'c:{comfort:5.2f}',
            f'terminal_reward:{terminal_reward:5.2f}'
        ]
        reward_debug = {
            'debug_texts': debug_texts
        }
        pdm_score = {
            "nac": no_at_fault_collisions,
            "dac": drivable_area_compliance,
            "ddc": driving_direction_compliance,
            "ep": ego_progress,
            "ttc": time_to_collision_within_bound,
            "c": comfort,
            "terminal_reward": terminal_reward
        }
        self.ego_vehicle.pdm_score = pdm_score
        return reward_debug
    def convert_relative_trajectories_bb(self,bb,start_idx,want_arr,real_arr,past_idx=None):
        if past_idx is None:
            past_idx = start_idx
        angle_diff = want_arr[start_idx][2] - real_arr[past_idx][2]
        agent_according_to_human = self.rotate(want_arr[start_idx,:2] - real_arr[past_idx,:2],-real_arr[past_idx,2])
        x,y,heading = bb
        xy = np.array([x,y])
        xy -= agent_according_to_human
        xy = self.rotate(xy,-angle_diff)
        heading -= angle_diff
        return xy[0],xy[1],heading
    def is_within_distance_ahead(self,target_location, max_distance, up_angle_th=60):
        distance = np.linalg.norm(target_location[0:2])
        if distance < 0.001:
            return True
        if distance > max_distance:
            return False
        # Bu x,y doğru olmayabilir
        x = target_location[0]
        y = target_location[1]
        angle = np.rad2deg(np.arctan2(y, x))
        return abs(angle) < up_angle_th
    def lbc_hazard_vehicle(self,frame,proximity_threshold=9.5,up_angle_th=np.pi/4):
        for i,name in enumerate(frame.annotations.names):
            if name != "vehicle":
                continue
            box = frame.annotations.boxes[i]
            bb = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
            )
            # -1 on time because we are rewarding previous frame
            x,y,heading = self.convert_relative_trajectories_bb(bb,self.ego_vehicle.time - 1,self.traj,self.ego_vehicle.human_trajectory,self.ego_vehicle.time -1 + 4) 
            same_heading = np.abs(heading) <= np.pi / 6 * 5 # Neden 150 yapmışlar anlamadım
            with_distance_ahead = self.is_within_distance_ahead(np.array([x,y]), proximity_threshold, up_angle_th=45)
            if same_heading and with_distance_ahead:
                return np.array([x,y,heading])
        return None
    def lbc_hazard_walker(self,frame,proximity_threshold=9.5,up_angle_th=np.pi/4):
        for i,name in enumerate(frame.annotations.names):
            if name != "pedestrian":
                continue
            box = frame.annotations.boxes[i]
            bbox = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
            )
            x,y,heading = self.convert_relative_trajectories_bb(bbox,self.ego_vehicle.time,self.traj,self.ego_vehicle.human_trajectory)
            dist = np.linalg.norm(np.array([x,y]))
            degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)
            if self.is_within_distance_ahead(np.array([x,y]), proximity_threshold, up_angle_th=degree):
                return np.array([x,y])
        return None
    def rotate(self, points, angle):
        """Rotate points by a given angle."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix.T
    def convert_relative_trajectories(self,points,start_idx,want_arr,real_arr,past_idx=None):
        if past_idx is None:
            past_idx = start_idx
        res = []
        angle_diff = want_arr[start_idx][2] - real_arr[past_idx][2]
        agent_according_to_human = self.rotate(want_arr[start_idx,:2] - real_arr[past_idx,:2],-real_arr[past_idx,2])
        for x,y,heading in points:
            xy = np.array([x,y])
            xy -= agent_according_to_human
            xy = self.rotate(xy,-angle_diff)
            heading -= angle_diff
            res.append(np.array([xy[0],xy[1],heading]))
        return np.array(res)
