import datetime

import numpy as np

from structs.app_enum import ActionType, PositionType


class TransactionManager:
    # ナンピンをしない（建玉を１単位しか持たない）売買管理クラス
    def __init__(
            self,
            reward_sell_buy=0.1,
            penalty_repay=-0.05,
            reward_pnl_scale=0.3,
            reward_hold=0.001,
            penalty_none=-0.001,
            penalty_rule=-1.0,
    ):
        self.reward_sell_buy = reward_sell_buy  # 約定ボーナスまたはペナルティ（買建、売建）
        self.penalty_repay = penalty_repay  # 約定ボーナスまたはペナルティ（返済）
        self.reward_pnl_scale = reward_pnl_scale  # 含み損益のスケール（含み損益✕係数）
        self.reward_hold = reward_hold  # 建玉を保持する報酬
        self.penalty_none = penalty_none  # 建玉を持たないペナルティ
        self.penalty_rule = penalty_rule  # 売買ルール違反

        # 売買ルール違反カウンター
        self.penalty_count = 0  # 売買ルール違反ペナルティを繰り返すとカウントを加算

        self.action_pre = ActionType.HOLD
        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.pnl_total = 0.0

        self.dict_transaction = self._init_transaction()
        self.code: str = '7011'
        self.unit: int = 1

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

    def _add_transaction(
            self,
            t: float,
            transaction: str,
            price: float,
            profit: float = np.nan,
    ):
        self.dict_transaction["注文日時"].append(self._get_datetime(t))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    @staticmethod
    def _get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def clearAll(self):
        """
        初期状態に設定
        :return:
        """
        self.resetPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0.0
        self.dict_transaction = self._init_transaction()

    def resetPosition(self):
        """
        ポジション（建玉）をリセット
        :return:
        """
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def setAction(self, action: ActionType, t: float, price: float) -> float:
        reward = 0.0
        if action == ActionType.HOLD:
            # ■■■ HOLD: 何もしない
            # 建玉があれば含み損益から報酬を付与、無ければ少しばかりの保持ボーナス
            reward += self._calc_reward_pnl(price)
            # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
            self.penalty_count = 0
        elif action == ActionType.BUY:
            # ■■■ BUY: 信用買い
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 買建 (LONG)
                self.position = PositionType.LONG
                self.price_entry = price
                # print(get_datetime(t), "買建", price)
                self._add_transaction(t, "買建", price)
                # 約定ボーナス付与（買建）
                reward += self.reward_sell_buy
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self._calc_reward_pnl(price)
                # ただし、建玉があるのに更に買建 (BUY) しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.SELL:
            # ■■■ SELL: 信用空売り
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 売建 (SHORT)
                self.position = PositionType.SHORT
                self.price_entry = price
                # print(get_datetime(t), "売建", price)
                self._add_transaction(t, "売建", price)
                # 約定ボーナス付与（売建）
                reward += self.reward_sell_buy
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self._calc_reward_pnl(price)
                # ただし、建玉があるのに更に売建しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.REPAY:
            # ■■■ REPAY: 建玉返済
            if self.position != PositionType.NONE:
                # ○○○ 建玉がある場合 ○○○
                if self.position == PositionType.LONG:
                    # 実現損益（売埋）
                    profit = price - self.price_entry
                    # print(get_datetime(t), "売埋", price, profit)
                    self._add_transaction(t, "売埋", price, profit)
                else:
                    # 実現損益（買埋）
                    profit = self.price_entry - price
                    # print(get_datetime(t), "買埋", price, profit)
                    self._add_transaction(t, "買埋", price, profit)
                # ポジション状態をリセット
                self.resetPosition()
                # 総収益を更新
                self.pnl_total += profit
                # 報酬に収益を追加
                reward += profit
                # 約定ペナルティ付与（返済）
                # reward += self.penalty_repay
                """
                commented by GPT-5
                実運用を見据えるなら「返済したら必ずペナルティ」という設計はちょっと違和感あります
                （決済はゴール動作なのでペナルティでなくてもいいかも？）。
                """
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # === 建玉がない場合 ===
                # 建玉がないのに建玉を返済しようとしたので売買ルール違反ペナルティを付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count
        else:
            raise ValueError(f"{action} is not defined!")

        self.action_pre = action
        return reward

    def _calc_reward_pnl(self, price: float) -> float:
        """
        含み損益に self.reward_pnl_scale を乗じた報酬を算出
        ポジションが無い場合は微小なペナルティを付与
        :param price:
        :return:
        """
        if self.position == PositionType.NONE:
            # PositionType.NONE に対して僅かなペナルティ
            return self.penalty_none
        else:
            reward = 0.0
            if self.position == PositionType.LONG:
                # 含み損益（買建）× 少数スケール
                reward += (price - self.price_entry) * self.reward_pnl_scale
            elif self.position == PositionType.SHORT:
                # 含み損益（売建）× 少数スケール
                reward += (self.price_entry - price) * self.reward_pnl_scale
            reward += self.reward_hold
            return reward
