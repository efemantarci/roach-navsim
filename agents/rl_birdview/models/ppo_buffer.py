from gym import spaces
import numpy as np
from typing import Optional, Generator, NamedTuple, Dict, List
import torch as th
from stable_baselines3.common.vec_env.base_vec_env import tile_images
import cv2
import time
from threading import Thread
import queue

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2

COLORS = [
    [46, 52, 54],
    [136, 138, 133],
    [255, 0, 255],
    [0, 255, 255],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 0],
    [255, 255, 255]
]


class PpoBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    old_mu: th.Tensor
    old_sigma: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    exploration_suggests: List[tuple]


class PpoBuffer():
    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
                 gae_lambda: float = 1, gamma: float = 0.99, n_envs: int = 1):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.reset()

        self.pos = 0
        self.full = False
        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.sample_queue = queue.Queue()

        # Render için
        self._pixels_per_meter = 1.5
        self._width = 192
        self.M_warp = self._get_warp_transform()

    def reset(self) -> None:
        self.observations = {}
        for k, s in self.observation_space.spaces.items():
            self.observations[k] = np.zeros((self.buffer_size, self.n_envs,)+s.shape, dtype=s.dtype)
        # int(np.prod(self.action_space.shape))
        self.actions = np.zeros((self.buffer_size, self.n_envs)+self.action_space.shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.mus = np.zeros((self.buffer_size, self.n_envs)+self.action_space.shape, dtype=np.float32)
        self.sigmas = np.zeros((self.buffer_size, self.n_envs)+self.action_space.shape, dtype=np.float32)
        self.exploration_suggests = np.zeros((self.buffer_size, self.n_envs), dtype=[('acc', 'U10'), ('steer', 'U10')])

        self.reward_debugs = [[] for i in range(self.n_envs)]
        self.terminal_debugs = [[] for i in range(self.n_envs)]

        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_value: th.Tensor, dones: np.ndarray) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
                # spinning up return calculation
                # self.returns[step] = self.rewards[step] + self.gamma * last_value * next_non_terminal
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
                # spinning up return calculation
                # self.returns[step] = self.rewards[step] + self.gamma * self.returns[step+1] * next_non_terminal
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # sb3 return
        self.returns = self.advantages + self.values

    def add(self,
            obs_dict: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray,
            mu: np.ndarray,
            sigma: np.ndarray,
            infos) -> None:

        for k, v in obs_dict.items():
            self.observations[k][self.pos] = v
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.mus[self.pos] = mu
        self.sigmas[self.pos] = sigma

        for i in range(self.n_envs):
            self.reward_debugs[i].append(infos[i]['reward_debug']['debug_texts'])
            self.terminal_debugs[i].append(infos[i]['terminal_debug']['debug_texts'])

            n_steps = infos[i]['terminal_debug']['exploration_suggest']['n_steps']
            if n_steps > 0:
                n_start = max(0, self.pos-n_steps)
                self.exploration_suggests[n_start:self.pos, i] = \
                    infos[i]['terminal_debug']['exploration_suggest']['suggest']

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def update_values(self, policy):
        for i in range(self.buffer_size):
            obs_dict = {}
            for k in self.observations.keys():
                obs_dict[k] = self.observations[k][i]
            values = policy.forward_value(obs_dict)
            self.values[i] = values

    def get(self, batch_size: Optional[int] = None) -> Generator[PpoBufferSamples, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        for tensor in ['actions', 'values', 'log_probs', 'advantages', 'returns',
                       'mus', 'sigmas', 'exploration_suggests']:
            self.__dict__['flat_'+tensor] = self.flatten(self.__dict__[tensor])
        self.flat_observations = {}
        for k in self.observations.keys():
            self.flat_observations[k] = self.flatten(self.observations[k])

        # spinning up: the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + np.finfo(np.float32).eps
        self.advantages = (self.advantages - adv_mean) / adv_std

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> PpoBufferSamples:
        def to_torch(x):
            return th.as_tensor(x).to(self.device)
            # return th.from_numpy(x.astype(np.float32)).to(self.device)

        obs_dict = {}
        for k in self.observations.keys():
            obs_dict[k] = to_torch(self.flat_observations[k][batch_inds])

        data = (self.flat_actions[batch_inds],
                self.flat_values[batch_inds],
                self.flat_log_probs[batch_inds],
                self.flat_mus[batch_inds],
                self.flat_sigmas[batch_inds],
                self.flat_advantages[batch_inds],
                self.flat_returns[batch_inds]
                )

        data_torch = (obs_dict,) + tuple(map(to_torch, data)) + (self.flat_exploration_suggests[batch_inds],)
        return PpoBufferSamples(*data_torch)

    @staticmethod
    def flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        # if len(shape) < 3:
        #     return arr.swapaxes(0, 1).reshape(shape[0] * shape[1])
        # else:
        return arr.reshape(shape[0] * shape[1], *shape[2:])

    def render(self):
        assert self.full, ''
        list_render = []

        _, _, c, h, w = self.observations['birdview'].shape
        vis_idx = np.array([0, 1, 2, 6, 10, 14])

        for i in range(self.buffer_size):
            im_envs = []
            for j in range(self.n_envs):

                masks = self.observations['birdview'][i, j, vis_idx, :, :] > 100

                im_birdview = np.zeros([h, w, 3], dtype=np.uint8)
                for idx_c in range(len(vis_idx)):
                    im_birdview[masks[idx_c]] = COLORS[idx_c]
                # ev_mask
                car_footprint = CarFootprint.build_from_rear_axle(
                    rear_axle_pose=StateSE2(0, 0, 0),
                    vehicle_parameters=get_pacifica_parameters(),
                )
                ev_mask = self._get_mask_from_actor_list([car_footprint.oriented_box], self.M_warp)
                im_birdview[ev_mask] = [255, 255, 255]
                im = np.zeros([h, w*2, 3], dtype=np.uint8)
                im[:h, :w] = im_birdview

                action_str = np.array2string(self.actions[i, j], precision=1, separator=',', suppress_small=True)
                state_str = np.array2string(self.observations['state'][i, j],
                                            precision=1, separator=',', suppress_small=True)

                reward = self.rewards[i, j]
                ret = self.returns[i, j]
                advantage = self.advantages[i, j]
                done = int(self.dones[i, j])
                value = self.values[i, j]
                log_prob = self.log_probs[i, j]

                txt_1 = f'v:{value:5.2f} p:{log_prob:5.2f} a{action_str}'
                im = cv2.putText(im, txt_1, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_2 = f'{done} {state_str}'
                im = cv2.putText(im, txt_2, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                txt_3 = f'rw:{reward:5.2f} rt:{ret:5.2f} a:{advantage:5.2f}'
                im = cv2.putText(im, txt_3, (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                for i_txt, txt in enumerate(self.reward_debugs[j][i] + self.terminal_debugs[j][i]):
                    im = cv2.putText(im, txt, (w, (i_txt+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                im_envs.append(im)

            big_im = tile_images(im_envs)
            list_render.append(big_im)

        return list_render

    def start_caching(self, batch_size):
        thread1 = Thread(target=self.cache_to_cuda, args=(batch_size,))
        thread1.start()

    def cache_to_cuda(self, batch_size):
        self.sample_queue.queue.clear()

        for rollout_data in self.get(batch_size):
            while self.sample_queue.qsize() >= 2:
                time.sleep(0.01)
            self.sample_queue.put(rollout_data)

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos
    def _get_warp_transform(self):
        bottom_left = [32,-32]#ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left = [32,32]#ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = [-32,32]#ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec
        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1],
                            [0, 0],
                            [self._width-1, 0]], dtype=np.float32)
        return cv2.getAffineTransform(src_pts, dst_pts)
    def _get_mask_from_actor_list(self, actor_list, M_warp, scale=1):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for bounding_box in actor_list:
            corners = bounding_box.all_corners()
            corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1] * scale] for corner in corners])
            corners_warped = cv2.transform(corners_in_pixel, M_warp)
            hull = cv2.convexHull(np.round(corners_warped).astype(np.int32))
            if hull.shape[1] != 1:
                hull = hull.reshape(-1, 1, 2)
            cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(bool)
    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * location.x #(location.x - self._world_offset[0])
        y = self._pixels_per_meter * location.y #(location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p