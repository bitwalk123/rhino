import gymnasium as gym
import numpy as np
import pandas as pd
import talib as ta
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


# TransactionManager はそのまま使える
class TransactionManager:
    def __init__(
            self,
            reward_rule=1.0,
            penalty_rule=-1.0,
            reward_pnl_scale=0.01,  # 含み損益のスケール（比率に対する係数）
            trade_cost_pct=0.0005,  # 約定ごとのコスト(0.05%)
            use_pct=True,
            clip_scale=5.0
    ):  # tanh スケール
        self.reward_rule = reward_rule
        self.penalty_rule = penalty_rule
        self.reward_pnl_scale = reward_pnl_scale
        self.trade_cost_pct = trade_cost_pct
        self.use_pct = use_pct
        self.clip_scale = clip_scale

        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0.0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def clearAll(self):
        self.clearPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0.0

    def has_position(self) -> bool:
        return self.price_entry > 0

    def _pct_pnl(self, price):
        if self.price_entry <= 0:
            return 0.0
        if self.position == PositionType.LONG:
            return (price - self.price_entry) / self.price_entry
        elif self.position == PositionType.SHORT:
            return (self.price_entry - price) / self.price_entry
        return 0.0

    def setAction(self, action: ActionType, price: float):
        raw_rule = 0.0
        # --- 簡素化のため: HOLD/BUY/SELL/REPAY に対する小さな定数配分 ---
        # ここは既存ロジックを置き換える/調整してください
        if self.has_position():
            if action == ActionType.REPAY:
                raw_rule += self.reward_rule
            elif action == self.action_pre:
                raw_rule += -0.1  # 同じ行動重複等の小ペナルティ
            else:
                raw_rule += 0.0
        else:
            if action in (ActionType.BUY, ActionType.SELL):
                raw_rule += self.reward_rule
            elif action == ActionType.REPAY:
                raw_rule += self.penalty_rule

        # PnL: if closing position, give realized pct pnl scaled; else give small scaled live pnl
        pnl_reward = 0.0
        if self.position != PositionType.NONE:
            pct = self._pct_pnl(price) if self.use_pct else (price - self.price_entry)
            if action == ActionType.REPAY:
                pnl_reward += pct * 1.0  # realized full pct
                self.pnl_total += pct
                # apply trade cost (percentage of entry price)
                pnl_reward -= self.trade_cost_pct
                self.clearPosition()
            else:
                pnl_reward += pct * self.reward_pnl_scale  # small live signal
        else:
            # opening position: set entry and apply cost
            if action == ActionType.BUY:
                self.position = PositionType.LONG
                self.price_entry = price
                pnl_reward -= self.trade_cost_pct
            elif action == ActionType.SELL:
                self.position = PositionType.SHORT
                self.price_entry = price
                pnl_reward -= self.trade_cost_pct

        raw_total = raw_rule + pnl_reward

        # クリッピング（過度な振れを防ぐ）
        clipped = np.tanh(raw_total / self.clip_scale) * self.clip_scale

        # 更新
        self.action_pre = action

        # for debugging, you may want to return tuple (clipped, raw_rule, pnl_reward)
        return clipped


class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ
        self.last_volume = df.iloc[0]["Volume"]
        self.current_step = 0
        # self.transman = TransactionManager()
        """
        # 調整イメージ
        self.transman = TransactionManager(
            reward_rule=0.5,  # ルール適合小さめ報酬
            penalty_rule=-1.0,  # ペナルティは控えめ
            reward_pnl_scale=0.05,  # 含み損益スケール強化
            trade_cost_pct=0.0,  # コスト一旦ゼロで学習を促す
            clip_scale=1.0  # 報酬を -1〜+1 の範囲に収める
        )
        # 調整例（もう少しPnL重視）
        self.transman = TransactionManager(
            reward_rule=0.1,  # ルール報酬ほぼ無視
            penalty_rule=-0.2,  # ペナルティも控えめ
            reward_pnl_scale=0.1,  # 含み損益をしっかり反映
            trade_cost_pct=0.0,  # まだコスト無効
            clip_scale=5.0  # 報酬幅をある程度残す
        )
        """
        # 提案するテスト設定（完全PnL主導）
        self.transman = TransactionManager(
            reward_rule=0.0,  # ルール無効化
            penalty_rule=0.0,  # ペナルティ無効化
            reward_pnl_scale=0.1,  # PnLを報酬の中心に
            trade_cost_pct=0.0,  # まずはコストなし
            clip_scale=20.0  # 報酬幅を残す
        )
        # obs: Price + ΔVolume + MA60 + STD60 + RSI60 + Z60 + one-hot(3)
        n_features = 1 + 1 + 4 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.transman.clearAll()
        self.last_volume = self.df.iloc[0]["Volume"]
        obs = self._get_observation()
        return obs, {}

    def step(self, n_action: int):
        # --- ウォームアップ期間は強制 HOLD ---
        if self.current_step < 60:
            action = ActionType.HOLD
        else:
            action = ActionType(n_action)

        reward = 0.0
        done = False

        price = self.df.at[self.current_step, "Price"]
        reward += self.transman.setAction(action, price)
        obs = self._get_observation()

        if self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1

        dict_info = {"pnl_total": self.transman.pnl_total}
        return obs, reward, done, False, dict_info

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = row["Price"]

        # ΔVolume
        current_volume = row["Volume"]
        delta_volume = current_volume - self.last_volume
        log_volume = np.log1p(max(delta_volume, 0)).astype(np.float32)
        self.last_volume = current_volume

        # 過去60ティックのデータを切り出し
        start = max(0, self.current_step - 59)
        window = self.df.iloc[start:self.current_step + 1]

        if len(window) >= 60:
            close = window["Price"].values.astype(np.float64)
            ma60 = ta.SMA(close, timeperiod=60)[-1]
            rsi60 = ta.RSI(close, timeperiod=60)[-1]
            std60 = window["Price"].rolling(60).std().iloc[-1]
            z60 = (price - ma60) / std60 if std60 > 0 else 0.0
        else:
            ma60, std60, rsi60, z60 = 0.0, 0.0, 0.0, 0.0

        # PositionType → one-hot
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)

        obs = np.array(
            [price, log_volume, ma60, std60, rsi60, z60],
            dtype=np.float32
        )
        obs = np.concatenate([obs, pos_onehot])
        return obs
