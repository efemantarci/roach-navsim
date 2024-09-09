import numpy as np
from nuplan.common.geometry.convert import absolute_to_relative_poses,relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2
from gym_navsim.utils.conversion import convert_absolute_to_relative_se2_array

class ValeoNoDetPx(object):

    def __init__(self, ego_vehicle, exploration_suggest=False, eval_mode=False):
        self.ego_vehicle = ego_vehicle
        self.agent_input = self.ego_vehicle.scene.get_agent_input().ego_statuses[-1]
        self._exploration_suggest = exploration_suggest
        self._last_lat_dist = 0.0
        self._min_thresh_lat_dist = 3.5
        
        
        self._eval_mode = eval_mode
    def get(self, timestamp):
        """
        # Done condition 1: vehicle blocked
        c_blocked = self._ego_vehicle.info_criteria['blocked'] is not None

        # Done condition 2: lateral distance too large
        ev_loc = self._ego_vehicle.vehicle.get_location()
        wp_transform = self._ego_vehicle.get_route_transform()
        d_vec = ev_loc - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
        lat_dist = np.abs(np.dot(np_wp_unit_right, np_d_vec))

        if lat_dist - self._last_lat_dist > 0.8:
            thresh_lat_dist = lat_dist + 0.5
        else:
            thresh_lat_dist = max(self._min_thresh_lat_dist, self._last_lat_dist)
        c_lat_dist = lat_dist > thresh_lat_dist + 1e-2
        self._last_lat_dist = lat_dist
        
        # Done condition 3: running red light
        c_run_rl = self._ego_vehicle.info_criteria['run_red_light'] is not None
        # Done condition 4: collision
        c_collision = self._ego_vehicle.info_criteria['collision'] is not None
        # Done condition 5: run stop sign
        if self._ego_vehicle.info_criteria['run_stop_sign'] is not None \
                and self._ego_vehicle.info_criteria['run_stop_sign']['event'] == 'run':
            c_run_stop = True
        else:
            c_run_stop = False
        """
        # Done condition 2: lateral distance too large
                
        abs_traj = relative_to_absolute_poses(StateSE2(*self.ego_vehicle.scene.frames[3].ego_status.ego_pose),[StateSE2(*x) for x in self.ego_vehicle.trajectory])  
        origin = abs_traj[-2]
        # ok kötü duruyor ama şimdilik böyle
        
        pdm_relative = convert_absolute_to_relative_se2_array(origin,self.ego_vehicle.route_abs)
        path_relative = convert_absolute_to_relative_se2_array(origin,np.array([[*se2] for se2 in abs_traj]))
        
        last_pdm_rel = pdm_relative[len(path_relative) - 4]
        last_traj_rel = path_relative[-1]
        lat_dist = np.abs(last_traj_rel[1] - last_pdm_rel[1])
        c_lat_dist = lat_dist > 6
        # if lat_dist - self._last_lat_dist > 0.8:
        #     thresh_lat_dist = lat_dist + 0.5
        # else:
        #     thresh_lat_dist = max(self._min_thresh_lat_dist, self._last_lat_dist)
        # c_lat_dist = lat_dist > thresh_lat_dist + 1e-2
        # self._last_lat_dist = lat_dist

        # Done condition 3 : distance from pdm too large
        dist = np.linalg.norm(last_pdm_rel - last_traj_rel)
        c_dist = dist > 10.0
        
        # Done condition 6: collision_px
        if self._eval_mode:
            c_collision_px = False
        else:
            c_collision_px = self.ego_vehicle.collision_px

        # Done condition 7: outside road
        c_outside_road = self.ego_vehicle.outside_road
        c_run_rl = self.ego_vehicle.run_rl
            
        finished = timestamp > 7
        #done = c_blocked or c_lat_dist or c_run_rl or c_collision or c_run_stop or c_collision_px or timeout
        crash = self.ego_vehicle.pdm_score["nac"] < 0.5
        infraction = self.ego_vehicle.pdm_score["dac"] < 0.5
        done = c_collision_px or crash or infraction or c_lat_dist or c_outside_road or c_run_rl or c_dist
        # terminal reward
        terminal_reward = 0.0
        if done:
            terminal_reward = -1.0
            
        if c_collision_px or c_run_rl or c_outside_road:
            terminal_reward -= np.linalg.norm(self.ego_vehicle.velocity)
        # terminal guide
        exploration_suggest = {
            'n_steps': 0,
            'suggest': ('', '')
        }
        if self._exploration_suggest:
            if c_collision_px:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('stop', '')
        debug_texts = [f"done : {done}"]
        # Böyle bir şey mi varmış
        infractions = {
            'run_red_light': c_run_rl,
            'collision_px': c_collision_px,
            'outside_road': c_outside_road,
            'dist': c_dist,
            'lat_dist': c_lat_dist,
            'crash': crash,
            'infraction': infraction
        }
        for k,v in infractions.items():
            if not v:
                continue
            debug_texts.append(f"{k} : {v}")
            print(f"{k} : {v}")
        done |= finished
        terminal_debug = {"exploration_suggest": exploration_suggest, "debug_texts": debug_texts}
        #return done, timeout, terminal_reward, terminal_debug
        return done, finished, terminal_reward, terminal_debug
    def rotate(self, points, angle):
        """Rotate points by a given angle."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix.T
