import numpy as np
from nuplan.common.geometry.convert import absolute_to_relative_poses,relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2
from gym_navsim.utils.conversion import convert_absolute_to_relative_se2_array

class ValeoNoDetPx(object):
    # Used for benchmarking. Lateral distance too large condition is removed.

    def __init__(self, ego_vehicle, exploration_suggest=False, eval_mode=False):
        self.ego_vehicle = ego_vehicle
        self.agent_input = self.ego_vehicle.scene.get_agent_input().ego_statuses[-1]
        self._exploration_suggest = exploration_suggest
        self._last_lat_dist = 0.0
        self._min_thresh_lat_dist = 3.5
        
        
        self._eval_mode = eval_mode
        """
        Bunlar ne bilmiyorum
        self._eval_time = 1200
        """
    def get(self, timestamp):
        
        # Done condition 6: collision_px
        if self._eval_mode:
            c_collision_px = False
        else:
            c_collision_px = self.ego_vehicle.collision_px

        # Done condition 7: outside road
        c_outside_road = self.ego_vehicle.outside_road
        c_run_rl = self.ego_vehicle.run_rl
            
        finished = timestamp > 7
        done = c_collision_px or c_outside_road or c_run_rl
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
