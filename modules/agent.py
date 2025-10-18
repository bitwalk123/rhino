import os

from PySide6.QtCore import QObject, Signal
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_new, get_ppo_model_path
from modules.agent_auxiliary import EpisodeLimitCallback, SaveBestModelCallback
from modules.env import TrainingEnv
from structs.res import AppRes


class PPOAgent(QObject):
    finishedTraining = Signal(str)
    finishedInferring = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self._stopping = False
        # self.total_timesteps = 1572864
        # self.total_timesteps = 100000
        self.total_timesteps = 20000

    def get_env(self, file: str, code: str) -> DummyVecEnv:
        # Excel ファイルをフルパスに
        path_excel = get_collection_path(self.res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)

        # 環境のインスタンスを生成
        env_raw = TrainingEnv(df)
        env_monitor = Monitor(env_raw, self.res.dir_log)  # Monitorの利用
        env_vec = DummyVecEnv([lambda: env_monitor])

        return env_vec

    def stop(self):
        """安全に終了させるためのフラグ"""
        self._stopping = True

    def train(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path, _ = get_ppo_model_path(self.res, code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
            try:
                model = RecurrentPPO.load(model_path, env, verbose=1)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = get_ppo_model_new(env)
        else:
            print(f"新規にモデルを作成します。")
            model = get_ppo_model_new(env)

        # モデルの学習
        model.learn(
            total_timesteps=int(1e8),
            callback=EpisodeLimitCallback(max_episodes=10)
        )

        print(f"モデルを {model_path} に保存します。")
        model.save(model_path)

        # 学習環境の解放
        env.close()
        self.finishedTraining.emit(file)

    def train_cherry_pick(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        model = get_ppo_model_new(env)

        # モデルの学習
        model_path, reward_path = get_ppo_model_path(self.res, code)
        callback = SaveBestModelCallback(
            save_path=model_path,
            reward_path=reward_path,
            should_stop=lambda: self._stopping
        )
        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        # 最後の取引履歴
        df_transaction = env.envs[0].env.getTransaction()
        print(df_transaction)
        print(f"損益: {df_transaction["損益"].sum():.1f} 円")

        # 学習環境の解放
        env.close()
        self.finishedTraining.emit(file)

    def infer(self, file: str, code: str):
        pass
        """
        # 推論専用のプログラム作成まで保留
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path, reward_path = self.get_model_path(self.res, code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
        else:
            print(f"モデルを {model_path} がありませんでした。")
            self.finishedInferring.emit()
            return
        model = RecurrentPPO.load(model_path, env, verbose=1)

        # 学習環境のリセット
        obs, _ = env.reset()
        lstm_state = None
        total_reward = 0.0

        # 推論の実行
        episode_over = False
        while not episode_over:
            # モデルの推論
            action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
            print(action)

            # 1ステップ実行
            # obs, reward, terminated, truncated, info = env.step(action)
            obs, rewards, dones, infos = env.step(action)

            # モデル報酬
            total_reward += rewards[0]

            # エピソード完了
            # episode_over = terminated or truncated
            episode_over = dones[0]

        # 学習環境の解放
        env.close()

        self.finishedInferring.emit()
        """
