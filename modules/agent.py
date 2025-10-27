import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path, get_ppo_model_new, get_trained_ppo_model_path
from modules.agent_auxiliary import ActionMaskWrapper
from modules.env import TrainingEnv
from structs.res import AppRes


class PPOAgentSB3:
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.env_raw = None

    def get_env(self, file: str, code: str) -> Monitor:
        # Excel ファイルをフルパスに
        path_excel = get_collection_path(self.res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)

        # 環境のインスタンスを生成
        self.env_raw = env_raw = TrainingEnv(df)
        env = ActionMaskWrapper(env_raw)
        # SB3の環境チェック（オプション）
        check_env(env, warn=True)

        env_monitor = Monitor(env, self.res.dir_log)  # Monitorの利用

        return env_monitor

    def train(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path = get_ppo_model_path(self.res, code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
            try:
                model = PPO.load(model_path, env, verbose=1)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = PPO("MlpPolicy", env, verbose=1)
        else:
            print(f"新規にモデルを作成します。")
            model = PPO("MlpPolicy", env, verbose=1)

        # モデルの学習
        model.learn(total_timesteps=100_000)

        # モデルの保存
        print(f"モデルを {model_path} に保存します。")
        model.save(model_path)

        # 学習環境の解放
        env.close()

    def infer(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path = get_trained_ppo_model_path(self.res, code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
        else:
            print(f"モデルを {model_path} がありませんでした。")
            return
        model = PPO.load(model_path, env, verbose=1)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

        df_transaction = pd.DataFrame(self.env_raw.trans_man.dict_transaction)
        print(df_transaction)
        print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

        # 学習環境の解放
        env.close()
