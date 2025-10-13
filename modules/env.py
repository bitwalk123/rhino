from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from modules.tamer import Tamer


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

        # 銘柄コード
        code = "7011"  # 現在のところは固定で良い

        # 売買管理＆特徴量生成クラス
        self.tamer = Tamer(code)

        # ウォームアップ期間
        # self.period = 60

        # 現在の行位置
        self.step_current = 0

        # 観測空間
        n_history, n_feature = self.tamer.getObsDim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_history, n_feature),
            dtype=np.float32
        )

        # アクション空間
        self.action_space = gym.spaces.Discrete(self.tamer.getActionSize())

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.step_current = 0
        obs = self.tamer.clearAll()

        # 観測値を返す
        return obs, {}

    def step(self, action: int):
        # データフレームの指定行の時刻と株価を取得
        t = self.df.at[self.step_current, "Time"]
        price = self.df.at[self.step_current, "Price"]
        volume = self.df.at[self.step_current, "Volume"]

        # アクション（取引）に対する報酬と観測値
        # truncated: 外部的な制限で終了（＝時間切れやステップ上限）
        obs, reward, truncated = self.tamer.setAction(action, t, price, volume)

        # 次のループへ進むか判定
        terminated = False  # 環境の内部ルールで終了（＝失敗や成功）
        if not truncated:
            if self.step_current >= len(self.df) - 1:
                # 建玉を持っていれば強制返済
                reward += self.tamer.forceRepay(t, price)
                truncated = True
            # データフレームを読み込む行を更新
            self.step_current += 1

        # info 辞書に総PnL
        info = {"pnl_total": self.tamer.getPnLTotal()}

        return obs, reward, terminated, truncated, info

    def getTransaction(self) -> pd.DataFrame:
        return self.tamer.getTransaction()
