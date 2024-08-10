import numpy as np
from importlib import import_module
from gym_navsim.core.ego_vehicle import EgoVehicle
class EgoVehicleHandler(object):
    def __init__(self, reward_configs, terminal_configs):
        self.ego_vehicles = {}
        self.info_buffers = {}
        self.reward_buffers = {}
        self.reward_handlers = {}
        self.terminal_handlers = {}
        self._reward_configs = reward_configs
        self._terminal_configs = terminal_configs
        self.n_scenes = 0
        self.pdm_score_total = 0
    # Task config yerine obs config vericem. Zaten sadece env_id yi kullanıyoruz
    def reset(self, obs_config,scene,split):
        for ev_id in obs_config:
            self.ego_vehicles[ev_id] = EgoVehicle(scene,split)

            self.reward_handlers[ev_id] = self._build_instance(
                self._reward_configs[ev_id], self.ego_vehicles[ev_id])
            self.terminal_handlers[ev_id] = self._build_instance(
                self._terminal_configs[ev_id], self.ego_vehicles[ev_id])

            self.reward_buffers[ev_id] = []
            self.info_buffers[ev_id] = {
                'collisions_layout': [],
                'collisions_vehicle': [],
                'collisions_pedestrian': [],
                'collisions_others': [],
                'red_light': [],
                'encounter_light': [],
                'stop_infraction': [],
                'encounter_stop': [],
                'route_dev': [],
                'vehicle_blocked': [],
                'outside_lane': [],
                'wrong_lane': []
            }
    @staticmethod
    def _build_instance(config, ego_vehicle):
        module_str, class_str = config['entry_point'].split(':')
        _Class = getattr(import_module('gym_navsim.core.task_actor.ego_vehicle.'+module_str), class_str)
        return _Class(ego_vehicle, **config.get('kwargs', {}))

    def apply_control(self, action_dict):
        for ev_id, action in action_dict.items():
            throttle = action[0]
            steer = action[1] # -1 left 0 ahead 1 right
            angle = np.arcsin(steer) / 2
            speed = 3 # Bunu sonra alıcam
            delta_x = throttle * np.cos(angle) * speed
            delta_y = throttle * np.sin(angle) * speed

            old_trajectory = self.terminal_handlers[ev_id].ego_vehicle.trajectory
            last_trajectory = old_trajectory[-1]
            added_trajectory = np.array([[last_trajectory[0] + delta_x, last_trajectory[1] + delta_y, last_trajectory[2] + angle]])
            new_trajectory = np.concatenate([old_trajectory, added_trajectory])
            self.terminal_handlers[ev_id].ego_vehicle.trajectory = new_trajectory
            self.reward_handlers[ev_id].ego_vehicle.trajectory = new_trajectory
            self.ego_vehicles[ev_id].steer = steer

    def tick(self, timestamp):
        reward_dict, done_dict, info_dict = {}, {}, {}

        for ev_id, ev in self.ego_vehicles.items():
            # info yok abi
            ev.time = timestamp
            done, timeout, terminal_reward, terminal_debug = self.terminal_handlers[ev_id].get(timestamp)
            reward, reward_debug = self.reward_handlers[ev_id].get(terminal_reward)

            reward_dict[ev_id] = reward
            done_dict[ev_id] = done
            info_dict[ev_id] = {}
            info_dict[ev_id]['timeout'] = timeout
            info_dict[ev_id]['reward_debug'] = reward_debug
            info_dict[ev_id]['terminal_debug'] = terminal_debug
            
            # accumulate into buffers
            self.reward_buffers[ev_id].append(reward)
            # save episode summary
            if done:
                info_dict[ev_id]['episode_event'] = self.info_buffers[ev_id]
                self.n_scenes += 1
                self.pdm_score_total += self.ego_vehicles[ev_id].pdm_score["score"]
                total_length = 0.5 # Burayı carladan çekiyordu :D Alt satırı takmayın
                completed_length = 0.5
                total_length = max(total_length, 0.001)
                completed_length = max(completed_length, 0.001)

                outside_lane_length = np.sum([x['distance_traveled']
                                              for x in self.info_buffers[ev_id]['outside_lane']]) / 1000
                wrong_lane_length = np.sum([x['distance_traveled']
                                            for x in self.info_buffers[ev_id]['wrong_lane']]) / 1000
                """
                if ev._endless:
                    score_route = completed_length
                """
                score_route = completed_length
                n_collisions_layout = int(len(self.info_buffers[ev_id]['collisions_layout']))
                n_collisions_vehicle = int(len(self.info_buffers[ev_id]['collisions_vehicle']))
                n_collisions_pedestrian = int(len(self.info_buffers[ev_id]['collisions_pedestrian']))
                n_collisions_others = int(len(self.info_buffers[ev_id]['collisions_others']))
                n_red_light = int(len(self.info_buffers[ev_id]['red_light']))
                n_encounter_light = int(len(self.info_buffers[ev_id]['encounter_light']))
                n_stop_infraction = int(len(self.info_buffers[ev_id]['stop_infraction']))
                n_encounter_stop = int(len(self.info_buffers[ev_id]['encounter_stop']))


                info_dict[ev_id]['episode_stat'] = {
                    'score_route': score_route,
                    'score_penalty': 0,
                    'score_composed': max(score_route*0, 0.0),
                    'length': len(self.reward_buffers[ev_id]),
                    'reward': np.sum(self.reward_buffers[ev_id]),
                    'timeout': float(42),
                    'is_route_completed': float(42),
                    'is_route_completed_nocrash': True,
                    'route_completed_in_km': completed_length,
                    'route_length_in_km': total_length,
                    'percentage_outside_lane': outside_lane_length / completed_length,
                    'percentage_wrong_lane': wrong_lane_length / completed_length,
                    'collisions_layout': n_collisions_layout / completed_length,
                    'collisions_vehicle': n_collisions_vehicle / completed_length,
                    'collisions_pedestrian': n_collisions_pedestrian / completed_length,
                    'collisions_others': n_collisions_others / completed_length,
                    'red_light': n_red_light / completed_length,
                    'light_passed': n_encounter_light-n_red_light,
                    'encounter_light': n_encounter_light,
                    'stop_infraction': n_stop_infraction / completed_length,
                    'stop_passed': n_encounter_stop-n_stop_infraction,
                    'encounter_stop': n_encounter_stop,
                    'route_dev': len(self.info_buffers[ev_id]['route_dev']) / completed_length,
                    'vehicle_blocked': len(self.info_buffers[ev_id]['vehicle_blocked']) / completed_length,
                    "number_of_scenes": self.n_scenes,
                    "avg_pdm_score": self.pdm_score_total / self.n_scenes
                }

        done_dict['__all__'] = all(done for obs_id, done in done_dict.items())
        return reward_dict, done_dict, info_dict

    def clean(self):
        for ev_id, ev in self.ego_vehicles.items():
            ev.clean()
        self.ego_vehicles = {}
        self.reward_handlers = {}
        self.terminal_handlers = {}
        self.info_buffers = {}
        self.reward_buffers = {}
    def _initialize_scene(self,token):
        self.scene = self.scene_loader.get_scene_from_token(token)
        self.ego_vehicle = EgoVehicle(self.scene)
        self.ego_vehicle.trajectory = None