import datetime

import numpy as np

from structs.app_enum import PositionType, ActionType


class TransactionManager:
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

        # ほんの僅かなルール適合報酬
        self.reward_comply_rule_small = +0.01

        # 僅かな HOLD 報酬
        self.reward_hold_small = +0.001

        # 僅かな HOLD ペナルティ
        self.penalty_hold_small = -0.001
        # 収益 0 の時の僅かなペナルティ
        self.penalty_profit_zero = -0.001
        # 収益がマイナスの時のペナルティ・レシオ（保留）
        self.penalty_ratio_profit_minus = 1.0

        # 取引ルール違反
        self.penalty_rule_transaction = -1.0
        # 取引ルール違反カウンター
        self.count_violate_rule_transaction = 0

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
                # 取引ルール適合
                # reward += self._comply_transaction_rule()
                # ほんの僅かな HOLD ペナルティ
                reward += self.penalty_hold_small
            elif action_type == ActionType.BUY:
                # 取引ルール適合
                reward += self._comply_transaction_rule()
                # =============================================================
                # 買建 (LONG)
                # =============================================================
                self.position = PositionType.LONG  # ポジションを更新
                self.price_entry = price + self.slippage  # 取得価格
                self._add_transaction(t, "買建", price)
            elif action_type == ActionType.SELL:
                # 取引ルール適合
                reward += self._comply_transaction_rule()
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = price - self.slippage  # 取得価格
                self._add_transaction(t, "売建", price)
            elif action_type == ActionType.REPAY:
                # 取引ルール違反
                reward += self._violate_transaction_rule()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        else:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが有る場合に取りうるアクションは HOLD, REPAY
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # 取引ルール適合
                # reward += self._comply_transaction_rule()
                # ほんの僅かな HOLD 報酬
                reward += self.reward_hold_small
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                reward += self._violate_transaction_rule()
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                reward += self._violate_transaction_rule()
            elif action_type == ActionType.REPAY:
                # 取引ルール適合
                reward += self._comply_transaction_rule()
                profit = self.getProfit(price)  # 実現損益
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
                if profit > 0:
                    # 呼び値で割って報酬を正規化
                    reward += profit / self.tickprice
                if profit < 0:
                    # 収益がマイナスの時は、少し大きめに強調
                    reward += profit / self.tickprice * self.penalty_ratio_profit_minus
                else:
                    # profit == 0 の時は僅かなペナルティ
                    reward += self.penalty_profit_zero
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
