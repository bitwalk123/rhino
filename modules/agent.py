import datetime
import os
import zipfile

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path, get_trained_ppo_model_path
from modules.agent_auxiliary import ActionMaskWrapper
from modules.env import TrainingEnv
from structs.res import AppRes


class PPOAgentSB3:
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.env_raw = None
        self.results = dict()
        self.results["reward"] = list()
        # ユーザー情報
        self.user_file = "user_data/history.csv"
        self.temp_file = "tmp/history.csv"
        # 設定値
        self.total_timesteps = 100_000

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

    def train(self, file: str, code: str, new_model: bool = False):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path = get_ppo_model_path(self.res, code)
        if not new_model and os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
            try:
                model = PPO.load(model_path, env, verbose=1)
                # ユーザー情報の読込
                with zipfile.ZipFile(model_path, 'r') as zipf:
                    if self.user_file in zipf.namelist():
                        with zipf.open(self.user_file) as f:
                            df_history = pd.read_csv(f)
                            self.prep_user_data(file, code, df_history)
                    else:
                        print(f"{self.user_file} は ZIP 内に存在しません。")
                        self.prep_user_data(file, code)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = PPO("MlpPolicy", env, verbose=1)
                self.prep_user_data(file, code, df_history)
        else:
            print(f"新規にモデルを作成します。")
            model = PPO("MlpPolicy", env, verbose=1)
            self.prep_user_data(file, code)

        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        # モデルの保存
        print(f"モデルを {model_path} に保存します。")
        model.save(model_path)
        # 保存した zip ファイルにユーザー情報を追加
        with zipfile.ZipFile(model_path, 'a') as zipf:
            zipf.write(self.temp_file, arcname=self.user_file)

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
            # 報酬分布作成用
            self.results["reward"].append(reward)

        # 取引内容
        self.results["transaction"] = self.env_raw.getTransaction()

        # 学習環境の解放
        env.close()

    def prep_user_data(self, file: str, code: str, df: pd.DataFrame = pd.DataFrame()):
        """
        モデルの zip ファイルにユーザ情報を盛り込むための準備
        """
        r = len(df)
        if r == 0:
            df = pd.DataFrame({"datetime": [], "file": [], "code": []})
        df.loc[r] = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file, code]
        # モデルを保存した後に追加するファイルを仮置き
        df.to_csv(self.temp_file, index=False)
