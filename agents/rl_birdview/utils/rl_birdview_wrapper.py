import gym
import numpy as np
import cv2

class RlBirdviewWrapper(gym.Wrapper):
    def __init__(self, env, benchmark=False,input_states=[], acc_as_action=False):
        assert len(env._obs_configs) == 1
        self._ev_id = list(env._obs_configs.keys())[0]
        self._input_states = input_states
        self._acc_as_action = acc_as_action
        self._render_dict = {}
        self.eval_mode = benchmark

        state_spaces = []   
        if 'speed' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['speed']['speed_xy'])
        if 'speed_limit' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['control']['speed_limit']) 
        if 'control' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['control']['throttle'])
            state_spaces.append(env.observation_space[self._ev_id]['control']['steer'])
            # Brake and gear are not implemented
            #state_spaces.append(env.observation_space[self._ev_id]['control']['brake'])
            #state_spaces.append(env.observation_space[self._ev_id]['control']['gear'])
        if 'acc_xy' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['acc_xy'])
        if 'vel_xy' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_xy'])
        if 'vel_ang_z' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_ang_z'])
        state_low = np.concatenate([s.low for s in state_spaces])
        state_high = np.concatenate([s.high for s in state_spaces])

        env.observation_space = gym.spaces.Dict(
            {'state': gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
             'birdview': env.observation_space[self._ev_id]['birdview']['masks']})
        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            # Code is implemented for this part. Acc_as_action must be true
            env.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            env.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        super(RlBirdviewWrapper, self).__init__(env)

    def reset(self):
        if self.eval_mode:
            for ev_id in self.env.ev_handler._terminal_configs:
                self.env.ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = True
            self.env.switch_scene_loader("test")
        else:
            for ev_id in self.env.ev_handler._terminal_configs:
                self.env.ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = False
            self.env.switch_scene_loader("trainval")

        self.obs_ma = self.env.reset()
        obs = self.process_obs(self.obs_ma[self._ev_id], self._input_states)

        self._render_dict['prev_obs'] = obs
        if "prev_im_render" not in self._render_dict:
            self._render_dict['prev_im_render'] = self.obs_ma[self._ev_id]['birdview']['rendered']
        #self._render_dict['prev_im_render'] = self.obs_ma[self._ev_id]['birdview']['rendered']
        return obs
    def set_scene(self,token):
        self.env.initialize_scene(token)
    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action, self._acc_as_action)}
        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)
        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)
        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]
        self._render_dict = {
            'timestamp': self.env.time,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        '''
        train render: used in train_rl.py
        '''
        """
        self._render_dict['action_value'] = self.action_value
        self._render_dict['action_log_probs'] = self.action_log_probs
        self._render_dict['action_mu'] = self.action_mu
        self._render_dict['action_sigma'] = self.action_sigma
        """
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview
        """
        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(render_dict['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(render_dict['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)
        
        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{render_dict["action_value"]:5.2f} p:{render_dict["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        """
        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im
    def process_obs(self,obs, input_states, train=True):

        state_list = []
        if 'speed' in input_states:
            state_list.append(obs['speed']['speed_xy'])
        if 'speed_limit' in input_states:
            state_list.append(obs['control']['speed_limit'])
        if 'control' in input_states:
            state_list.append(obs['control']['throttle'])
            state_list.append(obs['control']['steer'])
            state_list.append(obs['control']['brake'])
            state_list.append(obs['control']['gear']/5.0)
        if 'acc_xy' in input_states:
            state_list.append(obs['velocity']['acc_xy'])
        if 'vel_xy' in input_states:
            state_list.append(obs['velocity']['vel_xy'])
        if 'vel_ang_z' in input_states:
            state_list.append(obs['velocity']['vel_ang_z'])

        state = np.concatenate(state_list)
        birdview = obs['birdview']['masks']
        self.rendered = obs['birdview']['rendered']
        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {
            'state': state.astype(np.float32),
            'birdview': birdview
        }
        return obs_dict
    @staticmethod
    def process_act(action, acc_as_action, train=True):
        """
        if not train:
            action = action[0]
        """
        """
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)
        """
        # Assuming acc_as_action is always true. So no brakes
        throttle, steer = action.astype(np.float64)
        throttle = np.clip(throttle, -1, 1)
        steer = np.clip(steer, -1, 1)
        return (throttle,steer)