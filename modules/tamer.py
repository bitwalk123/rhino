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
        # 取引回数の上限
        self.n_transaction_max = 100

        # 取引管理用インスタンス
        self.trans_man = TransactionManager(code)
        self.obs_man = ObservationManager()

    def clearAll(self) -> np.ndarray:
        self.trans_man.clear()
        return self.obs_man.getObsReset()

    def forceRepay(self, t: float, price: float) -> float:
        return self.trans_man.forceRepay(t, price)

    @staticmethod
    def getActionSize() -> int:
        return len(ActionType)

    def getObsDim(self) -> tuple[int, int]:
        return self.obs_man.getObsDim()

    def getPnLTotal(self) -> float:
        return self.trans_man.pnl_total

    def getPosition(self) -> PositionType:
        return self.trans_man.position

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.trans_man.dict_transaction)

    def setAction(self, action: int, t: float, price: float, volume: float) -> tuple:
        # 報酬の評価
        reward = self.trans_man.evalReward(action, t, price)
        n_transactions = self.trans_man.getNumberOfTransactions()
        # 観測（報酬の評価が先）
        obs = self.obs_man.getObs(
            price,  # 株価
            volume,  # 出来高
            self.trans_man.getProfit(price),  # 含み益
            (self.n_transaction_max - n_transactions) / self.n_transaction_max,  # 正規化した残り取引回数
            self.getPosition()  # ポジション
        )
        if n_transactions >= self.n_transaction_max:
            truncated = True
        else:
            truncated = False
        return obs, reward, truncated
