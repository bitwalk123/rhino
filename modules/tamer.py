import pandas as pd

from modules.transman import TransactionManager
from structs.app_enum import ActionType, PositionType


class Tamer:
    """
    ナンピンをしない（建玉を１単位しか持たない）売買システムで
    特徴量を算出して報酬を最大化する調教クラス
    """

    def __init__(self, code: str):
        # 取引管理用インスタンス
        self.transman = TransactionManager(code)

    def clearAll(self):
        self.transman.clear()

    @staticmethod
    def getActionSize() -> int:
        return len(ActionType)

    def getPnLTotal(self) -> float:
        return self.transman.pnl_total

    def getPosition(self) -> PositionType:
        return self.transman.position

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.transman.dict_transaction)

    def setAction(self, action: int, t: float, price: float, volume: float) -> float:
        # 報酬の評価
        reward = self.transman.eval_reward(action, t, price)

        return reward
