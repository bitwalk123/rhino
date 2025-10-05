import numpy as np
import pandas as pd

from modules.obsman import ObservationManager
from modules.transman import TransactionManager
from structs.app_enum import ActionType, PositionType


class Tamer:
    """
    ナンピンをしない（建玉を１単位しか持たない）売買システムで
    特徴量を算出して報酬を最大化する調教クラス
    """

    def __init__(self, code: str):
        # 取引管理用インスタンス
        self.trans_man = TransactionManager(code)
        self.obs_man = ObservationManager()

    def clearAll(self) -> np.ndarray:
        self.trans_man.clear()
        return self.obs_man.getObsReset()

    @staticmethod
    def getActionSize() -> int:
        return len(ActionType)

    def getObsSize(self) -> int:
        return self.obs_man.getObsSize()

    def getPnLTotal(self) -> float:
        return self.trans_man.pnl_total

    def getPosition(self) -> PositionType:
        return self.trans_man.position

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.trans_man.dict_transaction)

    def setAction(self, action: int, t: float, price: float, volume: float) -> tuple:
        # 報酬の評価
        reward = self.trans_man.evalReward(action, t, price)
        # 観測（報酬の評価が先）
        obs = self.obs_man.getObs(price, volume, self.getPosition())

        return obs, reward
