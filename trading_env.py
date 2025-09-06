import gymnasium as gym
import numpy as np
from enum import Enum

import pandas as pd


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


class TransactionManager:
    def __init__(self):
        self.reward_none = 0.0
        self.reward_pnl_patio = 0.01
        self.penalty_avg_down = -1.0

        self.position = PositionType.NONE
        self.entry_price = 0.0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.entry_price = 0.0

    def setAction(self, action: ActionType, price: float) -> float:
        if action == ActionType.HOLD:
            if self.position == PositionType.LONG:
                reward = (price - self.entry_price) * self.reward_pnl_patio
            elif self.position == PositionType.SHORT:
                reward = (self.entry_price - price) * self.reward_pnl_patio
            else:
                reward = self.reward_none
        elif self.position == PositionType.LONG:
            if action == ActionType.REPAY:
                reward = price - self.entry_price
                self.clearPosition()
            else:
                reward = self.penalty_avg_down
        elif self.position == PositionType.SHORT:
            if action == ActionType.REPAY:
                reward = self.entry_price - price
                self.clearPosition()
            else:
                reward = self.penalty_avg_down
        elif self.position == PositionType.NONE:
            if action == ActionType.BUY:
                self.position = PositionType.LONG
                self.entry_price = price
            elif action == ActionType.SELL:
                self.position = PositionType.SHORT
                self.entry_price = price
            elif action == ActionType.REPAY:
                # ペナルティ要検討
                pass
            else:
                # ありえないケース
                pass
            reward = self.reward_none
        else:
            reward = self.reward_none

        return reward


class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.transman = TransactionManager()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1] - 1,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.transman.clearPosition()
        obs = self._get_observation()
        return obs, {}

    def step(self, action: ActionType):
        reward = 0.0
        done = False

        price = self.df.at[self.current_step, "Price"]
        reward += self.transman.setAction(action, price)
        obs = self._get_observation()
        if self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1

        """
        obs: 行動後の状態（＝現在の観測）
        reward: そのステップで得られた報酬
        terminated: エピソードが「自然終了」したか（例：目標達成、時間切れ）
        truncated: エピソードが「強制終了」されたか（例：最大ステップ数到達）
        info: デバッグやログ用の追加情報（辞書型）
        """
        return obs, reward, done, False, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = row.drop("Time").values.astype(np.float32)
        return obs
