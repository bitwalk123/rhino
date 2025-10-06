from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from modules.tamer import Tamer
from structs.app_enum import ActionType, PositionType


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

        # 銘柄コード
        code = "7011"

        # 売買管理クラス
        self.tamer = Tamer(code)

        # ウォームアップ期間
        self.period = 60

        # 現在の行位置
        self.step_current = 0

        # 観測空間
        # n_features = len(self.cols_features) + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.tamer.getObsSize(),),
            dtype=np.float32
        )

        # アクション空間
        self.action_space = gym.spaces.Discrete(self.tamer.getActionSize())

    def _get_action_mask(self) -> np.ndarray:
        """
        行動マスク
        [HOLD, BUY, SELL, REPAY]
        :return:
        """
        # if self.step_current < self.period:
        # ウォーミングアップ期間
        #    return np.array([1, 0, 0, 0], dtype=np.int8)  # 強制HOLD
        if self.tamer.getPosition() == PositionType.NONE:
            # 建玉なし
            return np.array([1, 1, 1, 0], dtype=np.int8)  # HOLD, BUY, SELL
        else:
            # 建玉あり
            return np.array([1, 0, 0, 1], dtype=np.int8)  # HOLD, REPAY

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.step_current = 0
        obs = self.tamer.clearAll()

        # 最初の観測値を取得
        # obs = self._get_observation()

        # 観測値と行動マスクを返す
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        # --- ウォームアップ期間 (self.period) は強制 HOLD ---
        #if self.step_current < self.period:
        #    action = ActionType.HOLD.value

        # データフレームの指定行の時刻と株価を取得
        t = self.df.at[self.step_current, "Time"]
        price = self.df.at[self.step_current, "Price"]
        volume = self.df.at[self.step_current, "Volume"]

        # アクション（取引）に対する報酬と観測値
        obs, reward = self.tamer.setAction(action, t, price, volume)

        # 次のループへ進むか判定
        terminated = False  # 環境の内部ルールで終了（＝失敗や成功）
        truncated = False  # 外部的な制限で終了（＝時間切れやステップ上限）
        if self.step_current >= len(self.df) - 1:
            truncated = True

        # データフレームを読み込む行を更新
        self.step_current += 1

        # info 辞書に総PnLと行動マスク
        info = {
            "pnl_total": self.tamer.getPnLTotal(),
            "action_mask": self._get_action_mask()
        }
        return obs, reward, terminated, truncated, info

    def getTransaction(self) -> pd.DataFrame:
        return self.tamer.getTransaction()
