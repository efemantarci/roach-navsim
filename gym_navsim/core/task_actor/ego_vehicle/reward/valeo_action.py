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
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2
from gym_navsim.utils.conversion import convert_absolute_to_relative_se2_array
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from shapely import affinity,intersection,Point
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
        pdm_route = self.ego_vehicle.route_abs
        abs_traj = relative_to_absolute_poses(StateSE2(*self.scene.frames[3].ego_status.ego_pose),[StateSE2(*x) for x in self.ego_vehicle.trajectory])  
        origin = abs_traj[-2]
        # ok kötü duruyor ama şimdilik böyle
        
        pdm_relative = convert_absolute_to_relative_se2_array(origin,self.ego_vehicle.route_abs)
        path_relative = convert_absolute_to_relative_se2_array(origin,np.array([[*se2] for se2 in abs_traj]))
        
        last_pdm_rel = pdm_relative[len(path_relative) - 4]
        last_traj_rel = path_relative[-1]
        lateral_distance = np.abs(last_traj_rel[1] - last_pdm_rel[1])
        r_position = -1.0 * np.abs(lateral_distance / 2.0)

        angle_difference = np.abs(last_traj_rel[2] - last_pdm_rel[2]) / np.pi
        r_rotation = -1.0 * angle_difference

        start_idx = self.scene.scene_metadata.num_history_frames + self.ego_vehicle.time - 1 - 1 # -1 comes because we are rewarding previous frame

        last_traj = abs_traj[-1]
        hazard_vehicle_loc = self.lbc_hazard_vehicle(self.scene.frames[start_idx],last_traj,proximity_threshold=9.5)
        hazard_ped_loc = self.lbc_hazard_walker(self.scene.frames[start_idx],last_traj, proximity_threshold=9.5)
        traffic_light_dist = self.get_traffic_light(self.scene.frames[start_idx],last_traj)

        desired_spd_veh = desired_spd_ped = desired_spd_rl = self._maximum_speed
        if hazard_vehicle_loc is not None:
            # -4 Yaptım
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2]) -4.0)
            desired_spd_veh = self._maximum_speed * np.clip(dist_veh, 0.0, 5.0)/5.0
            #print("Hazard vehicle detected",hazard_vehicle_loc,"Desired speed:",desired_spd_veh)
        if hazard_ped_loc is not None:
            # Bunu da -6 yerine -3 yaptım
            dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2])-3.0)
            desired_spd_ped = self._maximum_speed * np.clip(dist_ped, 0.0, 5.0)/5.0
            #print("Hazard pedestrian detected",hazard_ped_loc,"Desired speed:",desired_spd_ped)
        if traffic_light_dist is not None:
            # -2.5 bu da
            dist_rl = max(0.0, traffic_light_dist-2.5)
            desired_spd_rl = self._maximum_speed * np.clip(dist_rl, 0.0, 5.0)/5.0
            #print("Red light detected",desired_spd_rl,"Desired speed:",desired_spd_rl)
        
        desired_speed = min(self._maximum_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl)
        ev_speed = np.linalg.norm(self.ego_vehicle.velocity)

        r_speed = 1.0 - np.abs(ev_speed - desired_speed) / self._maximum_speed

        reward_debug["debug_texts"].append(f'r_position:{r_position:5.2f}')
        reward_debug["debug_texts"].append(f'r_rotation:{r_rotation:5.2f}')
        reward_debug["debug_texts"].append(f'r_action:{r_action:5.2f}')
        reward_debug["debug_texts"].append(f'r_speed:{r_speed:5.2f}')

        #reward = self.ego_vehicle.pdm_score["ep"] + terminal_reward + r_position + r_rotation + r_action
        reward = terminal_reward + r_position + r_rotation + r_action + r_speed
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
            "terminal_reward": terminal_reward,
            "score": score
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
        distance = np.linalg.norm(target_location[:2])
        if distance < 0.001:
            return True
        if distance > max_distance:
            return False
        # Bu x,y doğru olmayabilir
        x = target_location[0]
        y = target_location[1]
        angle = np.rad2deg(np.arctan2(y, x))
        return abs(angle) < up_angle_th
    def lbc_hazard_vehicle(self,frame,abs_agent_pos,proximity_threshold=9.5,up_angle_th=np.pi/4):
        for i,name in enumerate(frame.annotations.names):
            if name != "vehicle":
                continue
            box = frame.annotations.boxes[i]
            bb = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
            )
            abs_box = relative_to_absolute_poses(StateSE2(*frame.ego_status.ego_pose),[StateSE2(*bb)])[0]
            same_heading = np.abs(abs_box.heading - abs_agent_pos.heading) <= np.pi / 6 * 5 # Neden 150 yapmışlar anlamadım
            dist = np.array([abs_box.x,abs_box.y]) - np.array([abs_agent_pos.x,abs_agent_pos.y]) 
            with_distance_ahead = self.is_within_distance_ahead(np.array(dist), proximity_threshold, up_angle_th=45)
            if same_heading and with_distance_ahead:
                return convert_absolute_to_relative_se2_array(abs_agent_pos,[abs_box.x,abs_box.y,abs_box.heading])
        return None
    def lbc_hazard_walker(self,frame,abs_agent_pos,proximity_threshold=9.5,up_angle_th=np.pi/4):
        for i,name in enumerate(frame.annotations.names):
            if name != "pedestrian":
                continue
            box = frame.annotations.boxes[i]
            bb = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
            )
            abs_box = relative_to_absolute_poses(StateSE2(*frame.ego_status.ego_pose),[StateSE2(*bb)])[0]
            dist = np.array([abs_box.x,abs_box.y]) - np.array([abs_agent_pos.x,abs_agent_pos.y]) 
            # Bu ne demek tam anlayamadım 
            degree = 162 / (np.clip(np.linalg.norm(dist), 1.5, 10.5)+0.3)
            if self.is_within_distance_ahead(dist, proximity_threshold, up_angle_th=degree):
                return convert_absolute_to_relative_se2_array(abs_agent_pos,[abs_box.x,abs_box.y,abs_box.heading])
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
    def get_traffic_light(self,frame,abs_agent_pos):
        traffic_lights = np.array(frame.traffic_lights)
        if len(traffic_lights) == 0:
            return None
        history_origin = StateSE2(*frame.ego_status.ego_pose)
        map_object_dict = self.scene.map_api.get_proximal_map_objects(
            point=history_origin.point,
            radius=max((64,64)),
            layers=[SemanticMapLayer.LANE_CONNECTOR,SemanticMapLayer.STOP_LINE],
        )
        for map_object in map_object_dict[SemanticMapLayer.LANE_CONNECTOR]:
            idxs = np.where(traffic_lights[:,0] == int(map_object.id))[0]
            if len(idxs) == 0:
                continue
            idx = idxs[0]
            if not traffic_lights[idx,1]: # Green light
                continue
            incoming = map_object.incoming_edges[0]
            if incoming.contains_point(abs_agent_pos):
                scaled = affinity.scale(incoming.polygon,xfact=1.05,yfact=1.05,origin="center")
                shape = intersection(scaled,map_object.polygon)
                return shape.distance(Point(abs_agent_pos.point))
        return None