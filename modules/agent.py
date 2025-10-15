import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from funcs.ios import get_excel_sheet
from modules.env import TrainingEnv
from structs.res import AppRes


class SaveBestModelCallback(BaseCallback):
    def __init__(
            self,
            save_path: str,
            reward_path: str,
            should_stop: Callable[[], bool],
            verbose=1
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.reward_path = reward_path
        self.best_mean_reward = self._load_best_reward()
        self.should_stop = should_stop

    def _load_best_reward(self):
        if os.path.exists(self.reward_path):
            with open(self.reward_path, "r") as f:
                return float(f.read())
        return -float("inf")

    def _save_best_reward(self, reward: float):
        with open(self.reward_path, "w") as f:
            f.write(str(reward))

    def _on_step(self) -> bool:
        if self.should_stop():
            print("🛑 Training interrupted by user.")
            return False  # 学習を中断

        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            if ep_reward > self.best_mean_reward:
                self.best_mean_reward = ep_reward
                # ■■■ self.model はどこで定義されている？ ■■■
                self.model.save(self.save_path)
                self._save_best_reward(ep_reward)
                if self.verbose > 0:
                    print(f"✅ New best reward: {ep_reward:.2f} → Model saved.")
        return True


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
        path_excel = self.get_source_path(file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)

        # 環境のインスタンスを生成
        env_raw = TrainingEnv(df)
        env_monitor = Monitor(env_raw, self.res.dir_log)  # Monitorの利用
        env_vec = DummyVecEnv([lambda: env_monitor])

        return env_vec

    def get_model_path(self, code: str) -> tuple[str, str]:
        model_path = os.path.join(self.res.dir_model, f"ppo_{code}.zip")
        reward_path = os.path.join(self.res.dir_model, f"best_reward_{code}.txt")
        return model_path, reward_path

    def get_source_path(self, file: str) -> str:
        path_excel = str(Path(os.path.join(self.res.dir_collection, file)).resolve())
        return path_excel

    def stop(self):
        """安全に終了させるためのフラグ"""
        self._stopping = True

    def train(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習済モデルを読み込む
        model_path, _ = self.get_model_path(code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
            model = RecurrentPPO.load(model_path, env, verbose=True)
        else:
            print(f"新規モデルを作成します。")
            # PPO モデルの生成
            # LSTM を含む方策ネットワーク MlpLstmPolicy を指定
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=True)

        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        print(f"モデルを {model_path} に保存します。")
        model.save(model_path)

        # 最後の取引履歴
        df_transaction = env.envs[0].env.getTransaction()
        print(df_transaction)
        print(f"損益: {df_transaction["損益"].sum():.1f} 円")

        # 学習環境の解放
        env.close()
        self.finishedTraining.emit(file)

    def train_cherry_pick(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # PPO モデルの生成
        # LSTM を含む方策ネットワーク MlpLstmPolicy を指定
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=True)

        # モデルの学習
        model_path, reward_path = self.get_model_path(code)
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
        model_path, reward_path = self.get_model_path(code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
        else:
            print(f"モデルを {model_path} がありませんでした。")
            self.finishedInferring.emit()
            return
        model = RecurrentPPO.load(model_path, env, verbose=True)

        # 学習環境のリセット
        # obs, _ = env.reset()
        obs = env.reset()
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
