import os
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

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
        # self.total_timesteps = 1572864
        self.total_timesteps = 393216
        # self.total_timesteps = 18432

    def get_env(self, file: str, code: str) -> TradingEnv:
        # Excel ファイルをフルパスに
        path_excel = self.get_source_path(file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        # 学習環境のインスタンスを生成
        env = TradingEnv(df)
        return env

    def get_model_path(self, code: str) -> str:
        return os.path.join(self.res.dir_model, f"ppo_{code}.zip")

    def get_source_path(self, file: str) -> str:
        path_excel = str(Path(os.path.join(self.res.dir_collection, file)).resolve())
        return path_excel

    def stop(self):
        """安全に終了させるためのフラグ"""
        self._stopping = True

    def train(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)
        env = Monitor(env, self.res.dir_log)  # Monitorの利用

        # PPO モデルの生成
        # LSTM を含む方策ネットワーク MlpLstmPolicy を指定
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=True)

        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        # モデルの保存
        model_path = self.get_model_path(code)
        model.save(model_path)
        print(f"モデルを {model_path} に保存しました。")

        # 学習環境の解放
        env.close()
        self.finishedTraining.emit()

    def infer(self, file: str, code: str):
        # 学習環境の取得
        env = self.get_env(file, code)

        # 学習環境のリセット
        obs, info = env.reset()
        lstm_state = None
        total_reward = 0.0

        # 学習済モデルを読み込む
        model_path = self.get_model_path(code)
        if os.path.exists(model_path):
            print(f"モデル {model_path} を読み込みます。")
        else:
            print(f"モデルを {model_path} がありませんでした。")
            self.finishedInferring.emit()
            return

        model = RecurrentPPO.load(model_path, env, verbose=True)

        # 推論の実行
        episode_over = False
        while not episode_over:
            # モデルの推論
            action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)

            # 1ステップ実行
            obs, reward, terminated, truncated, info = env.step(int(action))

            # モデル報酬
            total_reward += reward

            # エピソード完了
            episode_over = terminated or truncated

        print("取引明細")
        print(env.getTransaction())

        print(f"--- テスト結果 ---")
        # モデル報酬（総額）
        print(f"モデル報酬（総額）: {total_reward:.2f}")
        if "pnl_total" in info.keys():
            print(f"最終的な累積報酬（1 株利益）: {info['pnl_total']:.2f}")

        # 学習環境の解放
        env.close()

        self.finishedInferring.emit()
