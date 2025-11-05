import datetime
from collections import deque
from enum import Enum
from typing import override

import gymnasium as gym
import numpy as np
import pandas as pd
import talib


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
        self.position = PositionType.NONE  # ポジション（建玉）
        self.price_entry = 0.0  # 取得価格
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self.init_transaction()  # 取引明細
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        """
        tanh が直線的に変化する [-1, 1] に収まるように割る因子
        ティックは最大 100 動くと仮定
        """
        self.factor_scale = 100.
        """
        含み益の場合に乗ずる比率
        """
        self.ratio_unrealized_profit = 0.01
        """
        確定損益が 0 の場合のペナルティ
        """
        self.penalty_zero_profit = -0.5
        """
        含み損益の保持のカウンター
        含み損益のインセンティブ・ペナルティ比率
        """
        self.count_unreal_profit = 0
        self.ratio_unreal_profit = 0.00005

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
        self.count_unreal_profit = 0

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
                self.price_entry = price  # 取得価格
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction(t, "買建", price)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = price  # 取得価格
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
                # =============================================================
                # 含み益
                # =============================================================
                profit = self.get_profit(price)
                reward += self.get_reward_from_profit(profit) * self.ratio_unrealized_profit
                if profit > 0:
                    self.count_unreal_profit += 1
                    reward += self.count_unreal_profit * self.ratio_unreal_profit
                elif profit < 0:
                    self.count_unreal_profit += 1
                    reward -= self.count_unreal_profit * self.ratio_unreal_profit

            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.REPAY:
                # =============================================================
                # 返済
                # =============================================================
                profit = self.get_profit(price)
                # 損益追加
                self.pnl_total += profit
                # 報酬
                if 0.0 <= profit <= 1.0:
                    reward += self.get_reward_from_profit(self.penalty_zero_profit)
                else:
                    reward += self.get_reward_from_profit(profit)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                if self.position == PositionType.LONG:
                    # 返済: 買建 (LONG) → 売埋
                    self.add_transaction(t, "売埋", price, profit)
                elif self.position == PositionType.SHORT:
                    # 返済: 売建 (SHORT) → 買埋
                    self.add_transaction(t, "買埋", price, profit)
                else:
                    raise TypeError(f"Unknown PositionType: {self.position}")
                # =============================================================
                # ポジション解消
                # =============================================================
                self.clear_position()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")

        return reward

    def forceRepay(self, t: float, price: float) -> float:
        reward = 0.0
        profit = self.get_profit(price)
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
        # 報酬
        reward += self.get_reward_from_profit(profit)
        # =====================================================================
        # ポジション解消
        # =====================================================================
        self.clear_position()

        return reward

    @staticmethod
    def get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def get_profit(self, price) -> float:
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

    def get_reward_from_profit(self, profit: float) -> float:
        # 報酬は呼び値で割る
        return np.tanh(profit / self.tickprice / self.factor_scale)

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getPL4Obs(self, price) -> float:
        """
        観測値用に、損益用の報酬と同じにスケーリングして含み損益を返す。
        """
        profit = self.get_profit(price)
        return self.get_reward_from_profit(profit)

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
    def __init__(self):
        # 調整用係数
        self.factor_mag = 20.
        self.unit = 100  # 最小取引単位（出来高）

        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.price_prev = 0.0  # １つ前の株価
        self.volume_prev = 0.0  # １つ前の出来高

        # キューを定義
        self.deque_price_060 = deque(maxlen=60)  # MA60
        self.deque_price_120 = deque(maxlen=120)  # MA120
        self.deque_price_300 = deque(maxlen=300)  # MA300

        # 観測数の取得
        self.n_feature = len(self.getObs())
        self.clear()

    def clear(self):
        # 特徴量算出のために保持する変数
        self.price_open: float = 0.0  # ザラバの始値
        self.price_prev: float = 0.0  # １つ前の株価
        self.volume_prev: float = 0.0  # １つ前の出来高
        # キューのクリア
        self.deque_price_060.clear()
        self.deque_price_120.clear()
        self.deque_price_300.clear()

    def func_moving_average(self, deque_price) -> float:
        return sum(deque_price) / len(deque_price)

    def func_price_delta(self, price: float) -> float:
        if self.price_prev == 0.0:
            price_delta = 0.0
        else:
            price_delta = (price - self.price_prev) / self.factor_mag

        self.price_prev = price
        return np.clip(price_delta, -1, 1)

    def func_price_ratio(self, price: float) -> float:
        if self.price_open == 0.0:
            # 寄り付いた最初の株価が基準価格
            self.price_open = price
            price_ratio = 0.0
        else:
            price_ratio = (price / self.price_open - 1.0) * self.factor_mag

        return price_ratio

    def func_ma_ratio(self, ma: float) -> float:
        if self.price_open == 0.0:
            ma_ratio = 0.0
        else:
            ma_ratio = (ma / self.price_open - 1.0) * self.factor_mag

        return ma_ratio

    def func_ratio_scaling(self, ratio: float) -> float:
        return np.clip((ratio - 1.0) * self.factor_mag, -1, 1)

    def func_volume_delta(self, volume: float) -> float:
        if self.volume_prev == 0.0:
            volume_delta = 0.0
        elif volume < self.volume_prev:
            volume_delta = 0.0
        else:
            x = (volume - self.volume_prev) / self.unit
            volume_delta = np.log1p(x)

        self.volume_prev = volume
        return np.tanh(volume_delta)

    def getObs(
            self,
            price: float = 0,  # 株価
            volume: float = 0,  # 出来高
            pl: float = 0,  # 含み損益
            count_hold: int = 0,  # HOLD 継続カウンタ
            position: PositionType = PositionType.NONE  # ポジション
    ) -> np.ndarray:
        list_feature = list()

        # 株価比率
        price_ratio = self.func_price_ratio(price)
        list_feature.append(price_ratio)

        # 株価差分
        # price_delta = self.func_price_delta(price)
        # list_feature.append(price_delta)

        # 累計出来高差分 / 最小取引単位
        # list_feature.append(self.func_volume_delta(volume))

        # キューへの追加
        self.deque_price_060.append(price)
        self.deque_price_120.append(price)
        self.deque_price_300.append(price)

        # 移動平均
        if price > 0:
            ma_060 = self.func_moving_average(self.deque_price_060)
            ma_120 = self.func_moving_average(self.deque_price_120)
            ma_300 = self.func_moving_average(self.deque_price_300)
        else:
            ma_060 = 0
            ma_120 = 0
            ma_300 = 0

        r_ma_060 = self.func_ma_ratio(ma_060)
        list_feature.append(r_ma_060)

        r_ma_120 = self.func_ma_ratio(ma_120)
        list_feature.append(r_ma_120)

        r_ma_300 = self.func_ma_ratio(ma_300)
        list_feature.append(r_ma_300)

        # 移動平均の差分
        ma_diff_1 = np.tanh((r_ma_060 - r_ma_120) * 2)
        list_feature.append(ma_diff_1)

        ma_diff_2 = np.tanh((r_ma_060 - r_ma_300) * 2)
        list_feature.append(ma_diff_2)

        ma_diff_3 = np.tanh((r_ma_120 - r_ma_300) * 2)
        list_feature.append(ma_diff_3)

        n = len(self.deque_price_300)

        # RSI: [-1, 1] に標準化
        n = len(self.deque_price_300)
        if n > 2:
            array_rsi = talib.RSI(
                np.array(self.deque_price_300, dtype=np.float64),
                timeperiod=n - 1
            )
            rsi = (array_rsi[-1] - 50.) / 50.
        else:
            rsi = 0.
        list_feature.append(rsi)

        # ---------------------------------------------------------------------
        # 含み損益
        # ---------------------------------------------------------------------
        list_feature.append(pl)

        # ---------------------------------------------------------------------
        # HOLD 継続カウンタ
        # ---------------------------------------------------------------------
        list_feature.append(np.tanh(count_hold / 5000.))

        # 一旦配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)

        # ---------------------------------------------------------------------
        # PositionType → one-hot (3) ［単位行列へ変換］
        # ---------------------------------------------------------------------
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)

        # arr_feature と pos_onehot を単純結合
        return np.concatenate([arr_feature, pos_onehot])

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()
        self.clear()
        return obs


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self):
        super().__init__()
        # ウォームアップ期間
        self.n_warmup: int = 60
        # 現在の行位置
        self.step_current: int = 0
        # 売買管理クラス
        self.trans_man = TransactionManager()
        # 観測値管理クラス
        self.obs_man = ObservationManager()
        # 観測空間
        n_feature = self.obs_man.n_feature
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature,),
            dtype=np.float32
        )
        # 行動空間
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

    def _get_tick(self) -> tuple[float, float, float]:
        ...

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.trans_man.dict_transaction)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        self.step_current = 0
        self.trans_man.clear()
        obs = self.obs_man.getObsReset()
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        ...


class TrainingEnv(TradingEnv):
    """
    環境クラス
    過去のティックデータを使った学習、推論用
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

    @override
    def _get_tick(self) -> tuple[float, float, float]:
        t: float = self.df.at[self.step_current, "Time"]
        price: float = self.df.at[self.step_current, "Price"]
        volume: float = self.df.at[self.step_current, "Volume"]
        return t, price, volume

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        過去のティックデータを使うことを前提とした step 処理
        """
        # --- ウォームアップ期間 (self.n_warmup) は強制 HOLD ---
        if self.step_current < self.n_warmup:
            action = ActionType.HOLD.value

        # データフレームからティックデータを取得
        t, price, volume = self._get_tick()
        # 報酬
        reward = self.trans_man.evalReward(action, t, price)
        # 観測値
        obs = self.obs_man.getObs(
            price,  # 株価
            volume,  # 出来高
            self.trans_man.getPL4Obs(price),  # 含み損益
            self.trans_man.count_unreal_profit,  # HOLD 継続カウンタ
            self.trans_man.position,  # ポジション
        )

        done = False
        truncated = False

        if self.step_current >= len(self.df) - 1:
            reward += self.trans_man.forceRepay(t, price)
            done = True
            truncated = True  # ← 時間切れによる終了を明示

        self.step_current += 1
        info = {
            "pnl_total": self.trans_man.pnl_total,
            "action_mask": self._get_action_mask()
        }

        return obs, reward, done, truncated, info
