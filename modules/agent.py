import os

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure

from modules.env import TrainingEnv


class PPOAgentSB3:
    def __init__(self):
        super().__init__()
        # 結果保持用辞書
        self.results = dict()
        # 設定値
        self.total_timesteps = 100_000


    def train(self, df: pd.DataFrame, path_model: str, log_dir: str, new_model: bool = False):
        custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])  # 出力形式を指定

        # 学習環境の取得
        env = TrainingEnv(df)
        # 学習済モデルを読み込む
        if not new_model and os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
            try:
                model = MaskablePPO.load(path_model, env, verbose=1)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = MaskablePPO("MlpPolicy", env, verbose=1)
        else:
            print(f"新規にモデルを作成します。")
            model = MaskablePPO("MlpPolicy", env, verbose=1)

        # ロガーを差し替え
        model.set_logger(custom_logger)

        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        # モデルの保存
        print(f"モデルを {path_model} に保存します。")
        model.save(path_model)

        # 学習環境の解放
        env.close()

    def infer(self, df: pd.DataFrame, path_model: str) -> bool:
        # 学習環境の取得
        env = TrainingEnv(df)

        # 学習済モデルを読み込む
        if os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
        else:
            print(f"モデルを {path_model} がありませんでした。")
            return False
        try:
            model = MaskablePPO.load(path_model, env, verbose=1)
        except ValueError as e:
            print(e)
            return False

        self.results["obs"] = list()
        self.results["reward"] = list()
        obs, _ = env.reset()
        done = False
        while not done:
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            # 観測値トレンド成用
            self.results["obs"].append(obs)
            # 報酬分布作成用
            self.results["reward"].append(reward)

        # 取引内容
        self.results["transaction"] = env.getTransaction()

        # 学習環境の解放
        env.close()

        return True
