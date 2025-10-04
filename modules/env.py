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
        # ウォームアップ期間
        self.period = 60
        # 特徴量の列名のリストが返る
        self.cols_features = self._add_features()
        # 現在の行位置
        self.current_step = 0
        # 売買管理クラス
        self.tamer = Tamer()
        # obs: len(self.cols_features) + one-hot(3)
        n_features = len(self.cols_features) + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def _add_features(self) -> list:
        """
        特徴量の追加
        :param period:
        :return:
        """
        list_features = list()

        # 調整用係数
        factor_ticker = 10  # 調整因子（銘柄別）
        unit = 100  # 最小取引単位

        # 最初の株価（株価比率の算出用）
        price_start = self.df["Price"].iloc[0]

        # 1. 株価差分
        colname = "PriceDelta"
        self.df[colname] = self.df["Price"].diff()
        list_features.append(colname)

        # 2. 株価比率
        colname = "PriceRatio"
        self.df[colname] = self.df["Price"] / price_start
        list_features.append(colname)

        # 3. 累計出来高差分 / 最小取引単位
        colname = "dVol"
        self.df[colname] = np.log1p(self.df["Volume"].diff() / unit) / factor_ticker
        list_features.append(colname)

        return list_features

    def _get_action_mask(self) -> np.ndarray:
        """
        行動マスク
        :return:
        """
        if self.current_step < self.period:
            # ウォーミングアップ期間
            return np.array([1, 0, 0, 0], dtype=np.int8)  # 強制HOLD
        if self.tamer.position == PositionType.NONE:
            # 建玉なし
            return np.array([1, 1, 1, 0], dtype=np.int8)  # HOLD, BUY, SELL
        else:
            # 建玉あり
            return np.array([1, 0, 0, 1], dtype=np.int8)  # HOLD, REPAY

    def _get_observation(self):
        if self.current_step >= self.period:
            features = self.df.iloc[self.current_step][self.cols_features]
        else:
            features = [0] * len(self.cols_features)
            """
            commented by GPT-5
            → ここも [0,...] にしているので、序盤の数十ステップが「完全にフラットな状態」になります。
            もし観測にノイズが少し欲しいなら、最初から EMA だけ計算して、差分系だけゼロにするなども検討できます
            （ただしこれはお好み次第）。
            """
        obs = np.array(features, dtype=np.float32)

        # PositionType → one-hot
        pos_onehot = np.eye(3)[self.tamer.position.value].astype(np.float32)
        obs = np.concatenate([obs, pos_onehot])

        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.tamer.clearAll()
        # 最初の観測値を取得
        obs = self._get_observation()
        # 観測値と行動マスクを返す
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        # --- ウォームアップ期間 (self.period) は強制 HOLD ---
        if self.current_step < self.period:
            action = ActionType.HOLD.value

        # データフレームの指定行の時刻と株価を取得
        t = self.df.at[self.current_step, "Time"]
        price = self.df.at[self.current_step, "Price"]
        volume = self.df.at[self.current_step, "Volume"]

        # アクション（取引）に対する報酬
        reward = self.tamer.setAction(action, t, price, volume)
        # 最初の観測値を取得
        obs = self._get_observation()

        # 次のループへ進むか判定
        done = False
        if self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1
        # info 辞書に総PnLと行動マスク
        info = {
            "pnl_total": self.tamer.pnl_total,
            "action_mask": self._get_action_mask()
        }
        # print(self.current_step, self.transman.action_pre, reward, done)
        return obs, reward, done, False, info
