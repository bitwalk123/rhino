import numpy as np

from structs.app_enum import PositionType


class ObservationManager:
    def __init__(self):
        # 調整用係数
        self.factor_ticker = 10.0  # 調整因子（銘柄別）
        self.unit = 100.0  # 最小取引単位

        self.price_init = 0.0
        self.price_prev = 0.0
        self.volume_prev = 0.0

    def _get_price_delta(self, price: float) -> float:
        if self.price_prev == 0:
            price_delta = 0
        else:
            price_delta = price - self.price_prev
        self.price_prev = price
        return price_delta

    def _get_price_ratio(self, price: float) -> float:
        if self.price_init == 0:
            self.price_init = price
            price_ratio = 1.0
        else:
            price_ratio = price / self.price_init
        return price_ratio

    def _get_volume_delta(self, volume: float):
        if self.volume_prev == 0.0:
            volume_delta = 0.0
        else:
            volume_delta = np.log1p((volume - self.volume_prev) / self.unit) / self.factor_ticker

        self.volume_prev = volume
        return volume_delta

    def getObs(
            self,
            price: float,  # 株価
            volume: float,  # 出来高
            profit: float,  # 含み益
            n_remain: int, # 残り取引回数
            position: PositionType  # ポジション
    ) -> np.ndarray:
        features = list()
        features.append(self._get_price_ratio(price))  # PriceRatio
        features.append(self._get_price_delta(price))  # PriceDelta
        # features.append(self._get_volume_delta(volume))  # VolumeDelta
        features.append(profit)  # 含み益
        features.append(n_remain)  # 残り取引回数
        arr_feature = np.array(features, dtype=np.float32)

        # PositionType → one-hot
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)
        obs = np.concatenate([arr_feature, pos_onehot])

        return obs

    def getObsReset(self) -> np.ndarray:
        n = self.getObsSize()
        return np.array([0] * n, dtype=np.float32)

    def getObsSize(self) -> int:
        obs = self.getObs(0.0, 0.0, 0.0,0, PositionType.NONE)
        return len(obs)
