import datetime
from enum import Enum

import gymnasium as gym
import numpy as np
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
    """
    売買管理クラス
    方策マスクでナンピンをしないことが前提
    """
    def __init__(self, code: str = '7011'):
        # #####################################################################
        # 取引関連
        # #####################################################################
        self.code: str = code  # 銘柄コード
        self.unit: int = 1  # 売買単位
        self.tickprice: float = 1.0  # 呼び値
        self.slippage = self.tickprice  # スリッページ

        self.position = PositionType.NONE  # ポジション（建玉）
        self.price_entry = 0.0  # 取得価格
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self._init_transaction()  # 取引明細

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ***** 取引ルール関係 *****
        # 取引ルール適合時の僅かな報酬
        self.reward_comply_rule_small = 0.
        # 取引ルール違反時のペナルティ
        self.penalty_rule_transaction = -0.
        # ***** HOLD 関係 *****
        # 建玉を持っている時の HOLD 報酬
        self.reward_hold_small = 0.
        # 建玉を持っていない時の HOLD ペナルティ
        self.penalty_hold = -0.
        # ***** 損益関係 *****
        # 建玉返済時に損益 0 の場合のペナルティ
        self.penalty_profit_zero = -0.

    def _add_transaction(self, t: float, transaction: str, price: float, profit: float = np.nan):
        self.dict_transaction["注文日時"].append(self._get_datetime(t))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    def _comply_transaction_rule(self) -> float:
        self.count_violate_rule_transaction = 0
        reward = self.reward_comply_rule_small
        return reward

    @staticmethod
    def _get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    @staticmethod
    def _init_transaction() -> dict:
        return {
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
        }

    def _violate_transaction_rule(self) -> float:
        self.count_violate_rule_transaction += 1
        return self.penalty_rule_transaction * self.count_violate_rule_transaction

    def clear(self):
        self.position = PositionType.NONE  # ポジション（建玉）
        self.price_entry = 0.0  # 取得価格
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self._init_transaction()  # 取引明細

    def forceRepay(self, t: float, price: float) -> float:
        profit = self.getProfit(price)
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            self._add_transaction(t, "売埋（強制返済）", price, profit)
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            self._add_transaction(t, "買埋（強制返済）", price, profit)
        else:
            pass
        # =====================================================================
        # 損益確定
        # =====================================================================
        reward = profit / self.tickprice  # 呼び値で割って報酬を正規化
        self.pnl_total += profit
        # =====================================================================
        # ポジション解消
        # =====================================================================
        self.position = PositionType.NONE
        self.price_entry = 0

        return reward

    def evalReward(self, action: int, t: float, price: float) -> float:
        action_type = ActionType(action)
        reward = 0.0

        if self.position == PositionType.NONE:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが無い場合に取りうるアクションは HOLD, BUY, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                pass
            elif action_type == ActionType.BUY:
                # =============================================================
                # 買建 (LONG)
                # =============================================================
                self.position = PositionType.LONG  # ポジションを更新
                self.price_entry = price + self.slippage  # 取得価格
                # =============================================================
                # 取引明細
                # =============================================================
                self._add_transaction(t, "買建", price)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = price - self.slippage  # 取得価格
                # =============================================================
                # 取引明細
                # =============================================================
                self._add_transaction(t, "売建", price)
            elif action_type == ActionType.REPAY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        else:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが有る場合に取りうるアクションは HOLD, REPAY
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                pass
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.REPAY:
                # 実現損益
                profit = self.getProfit(price)
                # =============================================================
                # 取引明細
                # =============================================================
                if self.position == PositionType.LONG:
                    # 返済: 買建 (LONG) → 売埋
                    self._add_transaction(t, "売埋", price, profit)
                elif self.position == PositionType.SHORT:
                    # 返済: 売建 (SHORT) → 買埋
                    self._add_transaction(t, "買埋", price, profit)
                else:
                    raise TypeError(f"Unknown PositionType: {self.position}")

                # =============================================================
                # 損益確定
                # =============================================================
                if profit == 0.0:
                    # profit == 0（損益 0）の時は僅かなペナルティ
                    pass
                else:
                    # 呼び値で割って報酬を正規化
                    reward += profit / self.tickprice

                self.pnl_total += profit

                # =============================================================
                # ポジション解消
                # =============================================================
                self.position = PositionType.NONE
                self.price_entry = 0
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")

        return reward

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getProfit(self, price) -> float:
        if self.position == PositionType.LONG:
            # ---------------------------------------------------------
            # 返済: 買建 (LONG) → 売埋
            # ---------------------------------------------------------
            return price - self.price_entry - self.slippage
        elif self.position == PositionType.SHORT:
            # ---------------------------------------------------------
            # 返済: 売建 (SHORT) → 買埋
            # ---------------------------------------------------------
            return self.price_entry - price + self.slippage
        else:
            return 0.0  # 実現損益


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
        self.transman = TransactionManager()
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
        # 特徴量の追加
        list_features = list()
        # 調整用係数
        factor_ticker = 10  # 調整因子（銘柄別）
        unit = 100  # 最小取引単位
        # 最初の株価（株価比率の算出用）
        price_start = self.df["Price"].iloc[0]
        # 1. 株価比率
        colname = "PriceRatio"
        self.df[colname] = self.df["Price"] / price_start
        list_features.append(colname)
        # 2. 累計出来高差分 / 最小取引単位
        colname = "dVol"
        self.df[colname] = np.log1p(self.df["Volume"].diff() / unit) / factor_ticker
        list_features.append(colname)
        return list_features

    def _get_action_mask(self) -> np.ndarray:
        # 行動マスク
        if self.current_step < self.period:
            # ウォーミングアップ期間
            return np.array([1, 0, 0, 0], dtype=np.int8)  # 強制 HOLD
        elif self.transman.position == PositionType.NONE:
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
        obs = np.array(features, dtype=np.float32)
        # PositionType → one-hot
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)
        obs = np.concatenate([obs, pos_onehot])
        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.transman.clear()
        obs = self._get_observation()
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, n_action: int):
        # --- ウォームアップ期間 (self.period) は強制 HOLD ---
        if self.current_step < self.period:
            action = ActionType.HOLD
        else:
            action = ActionType(n_action)
        reward = 0.0
        done = False
        t = self.df.at[self.current_step, "Time"]
        price = self.df.at[self.current_step, "Price"]
        reward += self.transman.setAction(action, t, price)
        obs = self._get_observation()
        if self.current_step >= len(self.df) - 1:
            done = True
        self.current_step += 1
        info = {"pnl_total": self.transman.pnl_total, "action_mask": self._get_action_mask()}
        return obs, reward, done, False, info

class TrainingEnv(TradingEnv):
    """
    環境クラス
    過去のティックデータを使った学習、推論用
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
