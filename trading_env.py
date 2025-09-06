import gymnasium as gym
import numpy as np
from enum import Enum


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


def is_valid_transition(action_prev, action_current, has_position):
    if has_position:
        if action_prev == ActionType.HOLD:
            return action_current in {ActionType.HOLD, ActionType.REPAY}
        elif action_prev == ActionType.BUY:
            return action_current in {ActionType.HOLD, ActionType.REPAY}
        elif action_prev == ActionType.SELL:
            return action_current in {ActionType.HOLD, ActionType.REPAY}
    else:
        if action_prev == ActionType.HOLD:
            return action_current in {ActionType.HOLD, ActionType.BUY, ActionType.SELL}
        elif action_prev == ActionType.REPAY:
            return action_current in {ActionType.BUY, ActionType.SELL, ActionType.HOLD}

    if action_prev == ActionType.REPAY and action_current == ActionType.REPAY:
        return False

    return True


class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.position = PositionType.NONE
        self.prev_action = ActionType.HOLD
        self.entry_price = 0.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1] - 1,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = PositionType.NONE
        self.prev_action = ActionType.HOLD
        self.entry_price = 0.0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = ActionType(action)
        reward = 0.0
        done = False

        if is_valid_transition(self.prev_action, action, self.has_position()):
            price = self.df.iloc[self.current_step]["Price"]

            if action == ActionType.BUY and self.position == PositionType.NONE:
                self.position = PositionType.LONG
                self.entry_price = price

            elif action == ActionType.SELL and self.position == PositionType.NONE:
                self.position = PositionType.SHORT
                self.entry_price = price

            elif action == ActionType.REPAY:
                if self.position == PositionType.LONG:
                    reward += price - self.entry_price
                elif self.position == PositionType.SHORT:
                    reward += self.entry_price - price
                self.position = PositionType.NONE
                self.entry_price = 0.0
        else:
            # 無効な遷移はペナルティ
            reward -= 1000.0

        self.prev_action = action
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = row.drop("Time").values.astype(np.float32)
        return obs

    def has_position(self):
        return self.position != PositionType.NONE
