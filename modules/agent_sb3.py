import gymnasium as gym
import numpy as np

class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_mask = info.get("action_mask", np.ones(self.env.action_space.n, dtype=np.int8))
        return obs

    def step(self, action):
        if self.action_mask[action] == 0:
            # 無効なアクションを選んだ場合、強制的に HOLD に置き換える
            action = 0  # ActionType.HOLD.value
        obs, reward, done, truncated, info = self.env.step(action)
        self.action_mask = info.get("action_mask", np.ones(self.env.action_space.n, dtype=np.int8))
        return obs, reward, done, truncated, info
