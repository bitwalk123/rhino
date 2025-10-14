from collections import deque

import numpy as np

from structs.app_enum import PositionType


class ObservationManager:
    def __init__(self):
        # 調整用係数
        self.factor_ticker = 10.0  # 調整因子（銘柄別）
        self.unit = 100.0  # 最小取引単位

        # 特徴量算出のために保持する変数
        self.price_init = 0.0  # ザラバの始値
        self.price_prev = 0.0  # １つ前の株価
        self.volume_prev = 0.0  # １つ前の出来高

        # 時系列の履歴数
        self.n_history = 30
        # FIFO バッファ（キュー）を作成
        self.deque_feature = deque(maxlen=self.n_history)
        # self.deque_feature の初期化（ゼロ埋め）
        self.initObs()
        # 特徴量の数
        self.n_feature = len(self.deque_feature[-1])
        print(f"(n_history, n_feature) = ({self.n_history}, {self.n_feature})")

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
        elif volume < self.volume_prev:
            """
            【稀に発生する警告】
            RuntimeWarning: invalid value encountered in log1p
            もしかするとリセットなどのタイミングで
            volume < self.volume_prev
            になるケースがあるかも！
            """
            volume_delta = 0.0
        else:
            x = (volume - self.volume_prev) / self.unit
            volume_delta = np.log1p(x) / self.factor_ticker

        self.volume_prev = volume
        return volume_delta

    def initObs(self):
        for i in range(self.n_history):
            self.getObs(0, 0, 0, 0, PositionType.NONE)

    def getObs(
            self,
            price: float,  # 株価
            volume: float,  # 出来高
            profit: float,  # 含み益
            n_remain: float,  # 残り取引回数
            position: PositionType  # ポジション
    ) -> np.ndarray:
        list_feature = list()

        # 1. PriceRatio
        list_feature.append(self._get_price_ratio(price))
        # 1. Price
        # list_feature.append(price)
        # 2. PriceDelta
        # list_feature.append(self._get_price_delta(price))
        # 2. VolumeDelta
        list_feature.append(self._get_volume_delta(volume))
        # 3. 含み益
        list_feature.append(profit)
        # 4. 残り取引回数
        list_feature.append(n_remain)

        # 一旦配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)

        # PositionType を単位行列へ変換
        # PositionType → one-hot (3)
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)

        # arr_feature と pos_onehot を単純結合
        features = np.concatenate([arr_feature, pos_onehot])

        # 特徴量をキューへ追加
        self.deque_feature.append(features)

        # キュー全体を配列にして返す
        return np.array(self.deque_feature, dtype=np.float32)

    def getObsDim(self) -> tuple[int, int]:
        return self.n_history, self.n_feature

    def getObsReset(self) -> np.ndarray:
        self.initObs()
        return np.array(self.deque_feature, dtype=np.float32)
