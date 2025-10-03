import os
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Signal
from stable_baselines3 import PPO

from funcs.ios import get_excel_sheet
from modules.env import TradingEnv
from structs.res import AppRes


class PPOAgent(QObject):
    finishedTraining = Signal()
    finishedInferring = Signal()

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
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        # 学習環境のインスタンスを生成
        env = TradingEnv(df)
        # PPO モデルの生成
        # 多層パーセプトロン (MLP) ベースの方策と価値関数を使う MlpPolicy を指定
        model = PPO("MlpPolicy", env, verbose=True)
        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        # 学習環境のリセット
        obs, info = env.reset()
        total_reward = 0.0

        # 推論の実行
        while True:
            # モデルの推論
            arr_action, _states = model.predict(obs, deterministic=True)
            action = arr_action.item()

            # 1ステップ実行
            obs, reward, done, truncated, info = env.step(action)

            # モデル報酬
            total_reward += reward

            # エピソード完了
            if done:
                break

        # 学習環境の解放
        env.close()


        print("取引明細")
        print(pd.DataFrame(env.transman.dict_transaction))

        print(f"--- テスト結果 ---")
        # モデル報酬（総額）
        print(f"モデル報酬（総額）: {total_reward:.2f}")
        if "pnl_total" in info.keys():
            print(f"最終的な累積報酬（1 株利益）: {info['pnl_total']:.2f}")

        self.finishedTraining.emit()
