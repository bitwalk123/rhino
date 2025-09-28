import os
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from stable_baselines3 import PPO

from funcs.ios import get_excel_sheet
from modules.env import TradingEnv
from structs.res import AppRes


class PPOAgent(QObject):
    finishedTraining = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self._stopping = False
        self.total_timesteps = 128000

    def get_source_path(self, file: str) -> str:
        path_excel = str(Path(os.path.join(self.res.dir_collection, file)).resolve())
        return path_excel

    def stop(self):
        """安全に終了させるためのフラグ"""
        self._stopping = True

    def train(self, file: str, code: str):
        # Excel ファイルをフルパスに
        path_excel = self.get_source_path(file)
        df = get_excel_sheet(path_excel, code)
        env = TradingEnv(df)
        model = PPO("MlpPolicy", env, verbose=True)
        model.learn(total_timesteps=self.total_timesteps)
        self.finishedTraining.emit()