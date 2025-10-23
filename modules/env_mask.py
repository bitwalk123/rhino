import datetime
from collections import deque
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
        self.code: str = code  # 銘柄コード
        # ---------------------------------------------------------------------
        # 取引関連
        # ---------------------------------------------------------------------
        self.unit: int = 1  # 売買単位
        self.tickprice: float = 1.0  # 呼び値
        # self.slippage = self.tickprice  # スリッページ
        self.slippage = 0  # スリッページ無し
        self.position = PositionType.NONE  # ポジション（建玉）
        self.price_entry = 0.0  # 取得価格
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self.init_transaction()  # 取引明細
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
        self.reward_hold_small = 0.0001
        # 建玉を持っていない時の HOLD ペナルティ
        self.penalty_hold = -0.
        # ***** 損益関係 *****
        # 建玉返済時に損益 0 の場合のペナルティ
        self.penalty_profit_zero = -0.1
        # 含み損益から報酬を算出する比
        self.reward_unrealized_profit_ratio = 0.1

    def add_transaction(self, t: float, transaction: str, price: float, profit: float = np.nan):
        self.dict_transaction["注文日時"].append(self.get_datetime(t))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    def clear(self):
        self.clear_position()
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self.init_transaction()  # 取引明細

    def clear_position(self):
        self.position = PositionType.NONE
        self.price_entry = 0

    def forceRepay(self, t: float, price: float) -> float:
        reward = 0
        profit = self.getProfit(price)
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction(t, "売埋（強制返済）", price, profit)
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction(t, "買埋（強制返済）", price, profit)
        else:
            # ポジション無し
            pass
        # 損益追加
        self.pnl_total += profit
        # -------------------------------------------------------------
        # 損益から報酬計算
        # -------------------------------------------------------------
        if profit == 0.0:
            # profit == 0（損益 0）の時は僅かなペナルティ
            # reward += self.penalty_profit_zero
            pass
        else:
            # 報酬は、呼び値で割って正規化
            reward += profit / self.tickprice
        # ---------------------------------------------------------------------
        # ポジション解消
        # ---------------------------------------------------------------------
        self.clear_position()

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
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction(t, "買建", price)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = price - self.slippage  # 取得価格
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction(t, "売建", price)
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
                # 建玉を持っている時の僅かな報酬
                # reward += self.reward_hold_small
                # 含み損益から報酬算出
                # profit = self.getProfit(price)
                # reward += profit / self.tickprice * self.reward_unrealized_profit_ratio
                pass
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.REPAY:
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                if self.position == PositionType.LONG:
                    # 返済: 買建 (LONG) → 売埋
                    price -= self.slippage
                    profit = self.getProfit(price)
                    self.add_transaction(t, "売埋", price, profit)
                elif self.position == PositionType.SHORT:
                    # 返済: 売建 (SHORT) → 買埋
                    price += self.slippage
                    profit = self.getProfit(price)
                    self.add_transaction(t, "買埋", price, profit)
                else:
                    raise TypeError(f"Unknown PositionType: {self.position}")
                # 損益追加
                self.pnl_total += profit
                # -------------------------------------------------------------
                # 損益から報酬計算
                # -------------------------------------------------------------
                if profit == 0.0:
                    # profit == 0（損益 0）の時は僅かなペナルティ
                    # reward += self.penalty_profit_zero
                    pass
                else:
                    # 報酬は、呼び値で割って正規化
                    reward += profit / self.tickprice
                # -------------------------------------------------------------
                # ポジション解消
                # -------------------------------------------------------------
                self.clear_position()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")

        return reward

    @staticmethod
    def get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getProfit(self, price) -> float:
        if self.position == PositionType.LONG:
            # ---------------------------------------------------------
            # 返済: 買建 (LONG) → 売埋
            # ---------------------------------------------------------
            return price - self.price_entry
        elif self.position == PositionType.SHORT:
            # ---------------------------------------------------------
            # 返済: 売建 (SHORT) → 買埋
            # ---------------------------------------------------------
            return self.price_entry - price
        else:
            return 0.0  # 実現損益

    @staticmethod
    def init_transaction() -> dict:
        return {
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
        }


class ObservationManager:
    def __init__(self, n_warmup: int):
        # 時系列の履歴数
        self.n_warmup = n_warmup
        # キューを作成
        self.deque_price = deque(maxlen=self.n_warmup)

        # 調整用係数
        self.factor_ticker = 10.0  # 調整因子（銘柄別）
        self.unit = 100  # 最小取引単位（出来高）

        # 特徴量算出のために保持する変数
        self.price_init = 0.0  # ザラバの始値
        self.price_prev = 0.0  # １つ前の株価
        self.volume_prev = 0.0  # １つ前の出来高

        self.n_feature = len(self.getObs())
        self.clear()

    def clear(self):
        # キューをクリア
        self.deque_price.clear()
        # 特徴量算出のために保持する変数
        self.price_init: float = 0.0  # ザラバの始値
        self.price_prev: float = 0.0  # １つ前の株価
        self.volume_prev: float = 0.0  # １つ前の出来高

    def func_price_ratio(self, price: float) -> float:
        if self.price_init == 0.0:
            self.price_init = price
            price_ratio = 1.0
        else:
            price_ratio = price / self.price_init
        return price_ratio

    def func_volume_delta(self, volume: float):
        if self.volume_prev == 0.0:
            volume_delta = 0.0
        elif volume < self.volume_prev:
            """
            【稀に発生する警告】
            RuntimeWarning: invalid value encountered in log1p
            """
            volume_delta = 0.0
        else:
            x = (volume - self.volume_prev) / self.unit
            volume_delta = np.log1p(x) / self.factor_ticker

        self.volume_prev = volume
        return volume_delta

    def getObs(
            self,
            price: float = 0,  # 株価
            volume: float = 0,  # 出来高
            position: PositionType = PositionType.NONE  # ポジション
    ) -> np.ndarray:
        list_feature = list()

        # 1. 株価比率
        list_feature.append(self.func_price_ratio(price))

        # 2. 累計出来高差分 / 最小取引単位
        list_feature.append(self.func_volume_delta(volume))

        # 一旦配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)

        # PositionType → one-hot (3) ［単位行列へ変換］
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)

        # arr_feature と pos_onehot を単純結合
        return np.concatenate([arr_feature, pos_onehot])

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()
        self.clear()
        return obs


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ
        # ウォームアップ期間
        self.n_warmup: int = 60
        # 現在の行位置
        self.step_current: int = 0
        # 売買管理クラス
        self.trans_man = TransactionManager()
        # 観測量管理クラス
        self.obs_man = ObservationManager(n_warmup=self.n_warmup)
        n_feature = self.obs_man.n_feature
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def _get_action_mask(self) -> np.ndarray:
        # 行動マスク
        if self.step_current < self.n_warmup:
            """
            ウォーミングアップ期間
            強制 HOLD
            """
            return np.array([1, 0, 0, 0], dtype=np.int8)
        elif self.trans_man.position == PositionType.NONE:
            """
            建玉なし
            取りうるアクション: HOLD, BUY, SELL
            """
            return np.array([1, 1, 1, 0], dtype=np.int8)
        else:
            """
            建玉あり
            取りうるアクション: HOLD, REPAY
            """
            return np.array([1, 0, 0, 1], dtype=np.int8)

    def reset(self, seed=None, options=None):
        self.step_current = 0
        self.trans_man.clear()
        obs = self.obs_man.getObsReset()
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        # --- ウォームアップ期間 (self.n_warmup) は強制 HOLD ---
        if self.step_current < self.n_warmup:
            action = ActionType.HOLD.value

        # データフレームからティックデータを取得
        t = self.df.at[self.step_current, "Time"]
        price = self.df.at[self.step_current, "Price"]
        volume = self.df.at[self.step_current, "Volume"]

        reward = self.trans_man.evalReward(action, t, price)
        obs = self.obs_man.getObs(
            price,  # 株価
            volume,  # 出来高
            self.trans_man.position,  # ポジション
        )

        done = False
        if self.step_current >= len(self.df) - 1:
            done = True

        self.step_current += 1
        info = {"pnl_total": self.trans_man.pnl_total, "action_mask": self._get_action_mask()}

        return obs, reward, done, False, info


class TrainingEnv(TradingEnv):
    """
    環境クラス
    過去のティックデータを使った学習、推論用
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
