import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from structs.res import AppRes


def get_ppo_model_new(env: DummyVecEnv) -> RecurrentPPO:
    # PPO モデルの生成
    # LSTM を含む方策ネットワーク MlpLstmPolicy を指定
    return RecurrentPPO("MlpLstmPolicy", env, verbose=1)


def get_ppo_model_path(res: AppRes, code: str) -> tuple[str, str]:
    model_path = os.path.join(res.dir_model, f"ppo_{code}.zip")
    reward_path = os.path.join(res.dir_model, f"best_reward_{code}.txt")
    return model_path, reward_path


def get_trained_ppo_model_path(res: AppRes, code: str) -> str:
    model_path = os.path.join(res.dir_model, "trained", f"ppo_{code}.zip")
    return model_path
