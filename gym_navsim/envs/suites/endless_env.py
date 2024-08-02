from gym_navsim.carla_multi_agent_env import NavsimEnv


class EndlessEnv(NavsimEnv):
    def __init__(self, token, obs_configs, reward_configs, terminal_configs):
        all_tasks = self.build_all_tasks()
        super().__init__(token, obs_configs, reward_configs, terminal_configs, all_tasks)

    @staticmethod
    def build_all_tasks():
        actor_configs_dict = {
            'ego_vehicles': {
                'hero': {'model': ''}
            }
        }
        route_descriptions_dict = {
            'ego_vehicles': {
                'hero': []
            }
        }
        endless_dict = {
            'ego_vehicles': {
                'hero': True
            }
        }
        all_tasks = []
        task = {
            'ego_vehicles': {
                'routes': route_descriptions_dict['ego_vehicles'],
                'actors': actor_configs_dict['ego_vehicles'],
                'endless': endless_dict['ego_vehicles']
            },
            'scenario_actors': {},
        }
        all_tasks.append(task)
        return all_tasks
